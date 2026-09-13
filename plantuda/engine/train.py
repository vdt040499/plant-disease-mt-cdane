"""The training loop shared by both arms.

One code path serves CDAN+E and CDAN+E + Mean Teacher; `use_mt` is the only
switch. Writing two loops would invite accidental divergence in RNG order,
model selection or augmentation, and any of those would invalidate the
comparison.

Four invariants keep the arms paired
------------------------------------
1. Construction order is ``FeatureExtractor -> Classifier -> CDANDisc`` in both
   arms. All three consume RNG at initialisation.
2. The teacher is built inside ``torch.random.fork_rng``. It constructs two
   more modules, so without the fork every subsequent source augmentation would
   be out of phase with the baseline.
3. ``seed_loader_epoch(seed, epoch)`` runs every epoch in both arms.
4. The three-view dataset generates its views inside its own forked RNG and is
   told the epoch through ``set_epoch``.

`scripts/rng_probe.py` measures invariants 1-4 directly, without training. Run
it before committing to a long run.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from plantuda.config import MeanTeacherConfig, NUM_CLASSES, TrainConfig
from plantuda.data.loaders import Loaders
from plantuda.engine.checkpoint import load_checkpoint, save_checkpoint
from plantuda.engine.evaluate import evaluate_accuracy
from plantuda.models.cdan import cdan_entropy_weight, multilinear_map
from plantuda.models.grl import GradRevLayer, domain_labels, get_lambda
from plantuda.models.mean_teacher import (
    consistency_weight,
    create_teacher,
    softmax_mse_loss,
    update_ema,
)
from plantuda.models.networks import CDANDisc, Classifier, FeatureExtractor
from plantuda.seeding import seed_loader_epoch, set_seed

ARM_NAMES = {False: "CDAN+E", True: "CDAN+E + Mean Teacher"}


@dataclass
class ArmResult:
    """Outcome of one run.

    Attributes:
        acc_selected: Accuracy of the selected model on the field test set, in
            percent. Selection uses source validation loss inside the window,
            and reads the teacher for the Mean Teacher arm.
        acc_student_final: Accuracy of the student's final weights. For the
            Mean Teacher arm, a large gap over `acc_selected` suggests the
            teacher's BatchNorm buffers are stale.
    """

    arm: str
    seed: int
    use_mt: bool
    acc_selected: float
    acc_student_final: float
    best_val_loss: float
    epochs_run: int

    def to_dict(self) -> dict:
        return asdict(self)


def train_arm(
    loaders: Loaders,
    *,
    use_mt: bool,
    device,
    train_cfg: TrainConfig = TrainConfig(),
    mt_cfg: MeanTeacherConfig = MeanTeacherConfig(),
    num_classes: int = NUM_CLASSES,
    ckpt_dir: Optional[str | os.PathLike] = None,
    pretrained: bool = True,
    verbose: bool = True,
) -> ArmResult:
    """Train one arm and evaluate it on the held-out field test set.

    Args:
        loaders: Built by `plantuda.data.build_loaders`.
        use_mt: False reproduces CDAN+E; True adds ``w(t) * L_cons`` and
            selects on the EMA teacher.
        ckpt_dir: Where to write ``<arm>_s<seed>_ckpt.pth``. None disables
            checkpointing, which is what the smoke tests want.
        pretrained: Load ImageNet weights into the backbone. Both arms must
            agree on this. Either value consumes the same amount of RNG, so it
            does not affect the pairing; tests pass False to skip the download.

    Returns:
        `ArmResult`.
    """
    tag = "mt" if use_mt else "cdane"
    seed = train_cfg.seed
    set_seed(seed)

    # (1) construction order, identical in both arms
    phi = FeatureExtractor(pretrained=pretrained).to(device)
    g = Classifier(phi.out_dim, num_classes).to(device)
    disc = CDANDisc(phi.out_dim, num_classes).to(device)
    grl = GradRevLayer().to(device)  # no parameters, consumes no RNG

    # (2) the teacher is built off the global RNG stream
    with torch.random.fork_rng(devices=[]):
        phi_t, g_t = create_teacher(phi, g, num_classes, device)

    optimizer = optim.AdamW(
        list(phi.parameters()) + list(g.parameters()) + list(disc.parameters()),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.max_epochs)

    ce_mean = nn.CrossEntropyLoss()
    ce_per_sample = nn.CrossEntropyLoss(reduction="none")

    ckpt_path = best_path = None
    start_epoch, step, best_loss, best_state = 1, 0, float("inf"), None
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{tag}_s{seed}_ckpt.pth")
        best_path = os.path.join(ckpt_dir, f"{tag}_s{seed}_best.pth")
        start_epoch, step, best_loss, best_state = load_checkpoint(
            ckpt_path, phi, g, disc, optimizer, scheduler, phi_t, g_t, device=device
        )
        if start_epoch > 1 and verbose:
            print(f"  [resume] continuing from epoch {start_epoch}")

    steps_per_epoch = loaders.steps_per_epoch
    total_steps = train_cfg.max_epochs * steps_per_epoch
    sel_phi, sel_g = (phi_t, g_t) if use_mt else (phi, g)
    lam = 0.0

    for epoch in range(start_epoch, train_cfg.max_epochs + 1):
        seed_loader_epoch(seed, epoch)                      # (3)
        loaders.target_three_view_dataset.set_epoch(epoch)  # (4)

        phi.train(); g.train(); disc.train()
        # The teacher must stay in train mode: its BatchNorm buffers are not
        # EMA-averaged and are only updated by its own forward passes.
        phi_t.train(); g_t.train()

        w_cons = consistency_weight(
            epoch, mt_cfg.cons_max, mt_cfg.cons_start, mt_cfg.cons_ramp_len
        )
        ep_cls = ep_cdan = ep_cons = 0.0
        target_hist = torch.zeros(num_classes, dtype=torch.long)

        for (src_img, src_lbl), ((t_plain, t_v1, t_v2), _) in zip(
            loaders.source_train, loaders.target_three_view
        ):
            src_img, src_lbl = src_img.to(device), src_lbl.to(device)
            t_plain = t_plain.to(device)

            lam = get_lambda(step, total_steps, train_cfg.grl_gamma)
            grl.set_lam(lam)
            step += 1
            optimizer.zero_grad()

            # ── CDAN+E branch ────────────────────────────────────────────────
            # The target input here is the un-augmented view, in both arms.
            src_feat = phi(src_img)
            tgt_feat = phi(t_plain)
            logits_s = g(src_feat)
            logits_t = g(tgt_feat)
            probs_s = torch.softmax(logits_s, dim=1)
            probs_t = torch.softmax(logits_t, dim=1)

            loss_cls = ce_mean(logits_s, src_lbl)

            # Detach the predictions (official CDAN) but keep the feature
            # gradient, so the adversarial signal still reaches phi via the GRL.
            src_mm = grl(multilinear_map(src_feat, probs_s.detach()))
            tgt_mm = grl(multilinear_map(tgt_feat, probs_t.detach()))

            w_src = cdan_entropy_weight(probs_s)
            w_src = w_src / w_src.sum()
            w_tgt = cdan_entropy_weight(probs_t)
            w_tgt = w_tgt / w_tgt.sum()
            d_src = (w_src * ce_per_sample(disc(src_mm), domain_labels(src_img.size(0), True, device))).sum()
            d_tgt = (w_tgt * ce_per_sample(disc(tgt_mm), domain_labels(t_plain.size(0), False, device))).sum()
            loss_cdan = (d_src + d_tgt) / 2.0

            # ── Mean Teacher branch ──────────────────────────────────────────
            # A separate pair of forward passes on the two augmented views, so
            # the CDAN+E branch above is bit-for-bit what the baseline runs.
            # Cost: two extra forward passes per iteration.
            if use_mt:
                t_v1, t_v2 = t_v1.to(device), t_v2.to(device)
                student_logits = g(phi(t_v1))
                with torch.no_grad():
                    teacher_logits = g_t(phi_t(t_v2))
                loss_cons = softmax_mse_loss(student_logits, teacher_logits) / t_v1.size(0)
                probe = teacher_logits
            else:
                # Without update_ema the teacher never leaves its initial state
                # (ImageNet backbone, random head), so the baseline measures its
                # class distribution on the student instead.
                loss_cons = torch.zeros((), device=device)
                probe = logits_t.detach()

            target_hist += torch.bincount(probe.argmax(1).cpu(), minlength=num_classes)

            total = loss_cls + loss_cdan + (w_cons * loss_cons if use_mt else 0.0)
            total.backward()
            optimizer.step()

            if use_mt:
                update_ema([phi, g], [phi_t, g_t], step, mt_cfg.ema_decay)

            ep_cls += loss_cls.item()
            ep_cdan += loss_cdan.item()
            ep_cons += loss_cons.item()

        scheduler.step()

        # ── Model selection on source validation ─────────────────────────────
        sel_phi.eval(); sel_g.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for img, lbl in loaders.source_val:
                img, lbl = img.to(device), lbl.to(device)
                logits = sel_g(sel_phi(img))
                val_loss_sum += ce_mean(logits, lbl).item()
                val_correct += (logits.argmax(1) == lbl).sum().item()
                val_total += lbl.size(0)
        val_loss = val_loss_sum / len(loaders.source_val)
        val_acc = val_correct / val_total

        if epoch >= train_cfg.min_select_epoch and val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                "phi": copy.deepcopy(sel_phi.state_dict()),
                "G": copy.deepcopy(sel_g.state_dict()),
            }
            if best_path:
                torch.save(best_state, best_path)

        is_log_epoch = epoch % train_cfg.log_every == 0 or epoch == train_cfg.max_epochs
        if verbose and is_log_epoch:
            share = target_hist.float() / max(target_hist.sum().item(), 1) * 100
            print(
                f"  [{tag}] ep {epoch:03d}/{train_cfg.max_epochs} "
                f"val_acc={val_acc * 100:5.1f}% lam={lam:.3f} w={w_cons:.2f} "
                f"| cls={ep_cls / steps_per_epoch:.3f} "
                f"cdan={ep_cdan / steps_per_epoch:.3f} "
                f"cons={ep_cons / steps_per_epoch:.4f} "
                f"| tgt%=[{share[0]:.0f},{share[1]:.0f},{share[2]:.0f}]"
            )
            if share.max().item() > 70.0:
                who = "teacher" if use_mt else "student"
                print(
                    f"     WARNING: {who} predictions are {share.max():.0f}% one class "
                    "- suspected class collapse"
                )

        if ckpt_path and is_log_epoch:
            save_checkpoint(
                ckpt_path,
                epoch=epoch,
                step=step,
                best_loss=best_loss,
                best_state=best_state,
                phi=phi,
                g=g,
                disc=disc,
                optimizer=optimizer,
                scheduler=scheduler,
                phi_teacher=phi_t,
                g_teacher=g_t,
            )

    # ── Final evaluation ─────────────────────────────────────────────────────
    if best_state is None:
        print(f"  WARNING [{tag}]: selection window never opened; using final weights.")
        best_state = {
            "phi": copy.deepcopy(sel_phi.state_dict()),
            "G": copy.deepcopy(sel_g.state_dict()),
        }

    eval_phi = FeatureExtractor(pretrained=False).to(device)
    eval_g = Classifier(eval_phi.out_dim, num_classes).to(device)
    eval_phi.load_state_dict(best_state["phi"])
    eval_g.load_state_dict(best_state["G"])

    acc_selected = evaluate_accuracy(eval_phi, eval_g, loaders.test, device) * 100
    acc_student_final = evaluate_accuracy(phi, g, loaders.test, device) * 100

    if use_mt and acc_student_final > acc_selected + 3.0:
        print(
            f"  WARNING: student ({acc_student_final:.2f}%) far ahead of teacher "
            f"({acc_selected:.2f}%) - suspected stale BatchNorm buffers"
        )

    return ArmResult(
        arm=ARM_NAMES[use_mt],
        seed=seed,
        use_mt=use_mt,
        acc_selected=acc_selected,
        acc_student_final=acc_student_final,
        best_val_loss=best_loss,
        epochs_run=train_cfg.max_epochs,
    )
