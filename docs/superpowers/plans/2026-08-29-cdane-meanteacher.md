# CDAN+E + Mean Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm nhánh CDAN+E + Mean Teacher vào `Plant_Disease_UDA_Final.ipynb` và chạy 5 seed cho cả nhánh này lẫn nhánh CDAN+E thuần, để kết luận có căn cứ thống kê xem consistency regularization có cải thiện CDAN+E trong bối cảnh UDA hay không.

**Architecture:** Một hàm `train_cdane(seed, use_mt)` duy nhất phục vụ cả hai nhánh — cờ `use_mt` là biến độc lập duy nhất. Teacher là EMA của student, không nhận gradient, được dùng làm mục tiêu consistency và làm model đánh giá. Notebook vẫn tự chứa (self-contained) như hiện tại; test chạy cục bộ bằng cách đọc thẳng source của cell từ file `.ipynb`, nên không có hai bản sao code.

**Tech Stack:** PyTorch, torchvision, numpy, scipy (paired t-test), pytest (chỉ để test cục bộ trên CPU). Notebook chạy trên Google Colab GPU.

## Global Constraints

- Backbone ResNet-18 (ImageNet init), 3 lớp: healthy / rust / scab.
- Optimizer AdamW, `lr=1e-3`, `weight_decay=0.01`, `CosineAnnealingLR(T_max=300)`. Teacher **không** nằm trong optimizer.
- `MAX_EPOCHS = 300`, `MIN_SEL = 250`, batch size 64 cho mọi loader.
- `EMA_DECAY = 0.99`, `CONS_MAX = 9.0`, `CONS_START = 100`, `CONS_RAMP_LEN = 50`.
- Consistency loss dùng đúng `softmax_mse_loss` của `github.com/milsever/plant-pathology/losses.py`.
- EMA dùng đúng `update_ema_variables` của `github.com/CuriousAI/mean-teacher`: `alpha = min(1 - 1/(global_step+1), ema_decay)`, chỉ duyệt `.parameters()`.
- Teacher ở `.train()` mode khi huấn luyện (để BN buffer cập nhật), `.eval()` khi validate/test, forward luôn bọc `torch.no_grad()`.
- Không sửa bất kỳ cell nào đang có trong notebook. Chỉ chèn cell mới ở vị trí ngay trước markdown `## 9 — Results Comparison`, và ở Task 7 mới nối thêm dòng vào bảng kết quả.
- Seed: 1, 2, 3, 4, 5. Mỗi run gọi `set_seed(s)` đầu tiên.

---

## Phát hiện quan trọng làm lệch khỏi spec

Đọc `split_ppd` (cell 13) cho thấy `dataset_B_target` là một `Subset` của `PlantDiseaseDataset` được tạo với **`transform=VAL_TRANSFORM`** (chỉ Resize, không augmentation) và `labeled=True`.

**Hệ quả:** nếu dùng thẳng `target_loader` hiện có cho Mean Teacher, hai view sẽ **giống hệt nhau từng pixel**, `L_cons` luôn bằng 0, và MT không làm gì cả. Đây là lỗi câm — training vẫn chạy 300 epoch bình thường rồi cho ra kết quả y hệt baseline.

**Cách xử lý:** dựng một dataset target mới dùng `TwoCropTransform(TRAIN_TRANSFORM)` và `labeled=False`, rồi cho **cả hai nhánh** dùng chung loader này (nhánh baseline chỉ lấy view 2, bỏ view 1). Nhờ vậy:

- Hai nhánh tiêu thụ RNG y hệt nhau → so sánh ghép cặp theo seed là thật.
- Biến độc lập duy nhất giữa hai nhánh đúng là sự có mặt của `L_cons`.

**Đánh đổi phải nêu rõ khi báo cáo:** baseline chạy lại sẽ **không** bằng đúng 87.28% của cell 42, vì ảnh target giờ có augmentation trong khi cell 42 thì không. Con số 87.28% từ nay chỉ là tham chiếu lịch sử, không phải mốc so sánh. Mốc so sánh là baseline 5 seed chạy lại trong Task 6.

---

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `Plant_Disease_UDA_Final.ipynb` | Sửa (chỉ chèn cell mới) | Toàn bộ pipeline; nguồn chân lý duy nhất của code |
| `tests/nbload.py` | Tạo | Đọc và exec source của cell từ `.ipynb` vào một namespace |
| `tests/test_mt_helpers.py` | Tạo | Test rampup, consistency weight, softmax MSE, EMA, cô lập gradient teacher, hành vi BN buffer |
| `tests/test_two_view.py` | Tạo | Test `TwoCropTransform` sinh hai view khác nhau |
| `.gitignore` | Tạo | Loại checkpoint, venv, cache |
| `requirements-dev.txt` | Tạo | torch/torchvision/pytest/numpy/scipy cho test cục bộ trên CPU |

**Quy ước tag cell:** mỗi cell mới bắt đầu bằng một dòng `# === CELL-TAG: <tên> ===`. Test dùng tag này để tìm và exec đúng cell, nên code chỉ tồn tại một bản duy nhất — trong notebook.

**Nơi chèn:** ngay trước cell markdown chứa chuỗi `## 9 — Results Comparison` (hiện là index 47; index sẽ dịch sau mỗi lần chèn, nên luôn định vị bằng nội dung, không bằng số).

**Phân công verify:** Task 1–4 verify được **cục bộ** trên CPU bằng pytest (tensor nhỏ, vài giây). Task 5–7 phải chạy trên **Colab GPU**. Mỗi task nêu rõ chạy ở đâu.

---

### Task 1: Bộ khung test đọc cell từ notebook

**Files:**
- Create: `.gitignore`
- Create: `requirements-dev.txt`
- Create: `tests/nbload.py`
- Test: `tests/test_nbload.py`

**Interfaces:**
- Consumes: không
- Produces: `nbload.load_cells(nb_path: str, contains: list[str], ns: dict | None = None) -> dict` — tìm các cell code có source chứa mỗi chuỗi trong `contains` (theo đúng thứ tự đã cho), exec chúng vào `ns`, trả về `ns`. Ném `LookupError` nếu một chuỗi không khớp cell nào hoặc khớp nhiều hơn một cell.

- [ ] **Step 1: Khởi tạo git**

Thư mục này chưa phải git repo. Kế hoạch cần commit sau mỗi task để 10 run dài có điểm quay lui.

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
git init
git branch -M main
```

- [ ] **Step 2: Tạo `.gitignore`**

```
.venv/
__pycache__/
.pytest_cache/
*.pyc
*.pth
fbr_cache/
```

- [ ] **Step 3: Tạo `requirements-dev.txt`**

```
torch
torchvision
numpy
scipy
pytest
```

- [ ] **Step 4: Dựng venv và cài đặt**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
python3 -m venv .venv
.venv/bin/pip install -q -r requirements-dev.txt
.venv/bin/python -c "import torch, scipy, pytest; print('torch', torch.__version__)"
```

Expected: in ra `torch 2.x.x` không lỗi. Đây là bản CPU, không cần GPU.

- [ ] **Step 5: Viết test thất bại**

Tạo `tests/test_nbload.py`:

```python
import os
import pytest
from nbload import load_cells, NB_PATH


def test_loads_feature_extractor_from_notebook():
    """Cell 32 của notebook định nghĩa FeatureExtractor — nbload phải exec được nó."""
    ns = {}
    exec(
        "import os, copy\n"
        "import numpy as np\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import torchvision.models as tv_models\n",
        ns,
    )
    load_cells(NB_PATH, contains=["class FeatureExtractor"], ns=ns)
    phi = ns["FeatureExtractor"](pretrained=False)
    assert phi.out_dim == 512


def test_raises_when_marker_not_found():
    with pytest.raises(LookupError):
        load_cells(NB_PATH, contains=["khong_ton_tai_chuoi_nay_dau"], ns={})
```

- [ ] **Step 6: Chạy test để xác nhận nó thất bại**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/ -v
```

Expected: FAIL với `ModuleNotFoundError: No module named 'nbload'`.

- [ ] **Step 7: Viết `tests/nbload.py`**

```python
"""Đọc và exec source của cell trong notebook, để test code notebook mà không nhân bản nó."""
import json
import os

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Plant_Disease_UDA_Final.ipynb",
)


def _code_cells(nb_path):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    return [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]


def load_cells(nb_path, contains, ns=None):
    """Exec vào `ns` các cell code khớp từng chuỗi trong `contains`, theo thứ tự.

    Mỗi chuỗi phải khớp đúng một cell. Khớp 0 hoặc >1 cell đều ném LookupError —
    im lặng chọn nhầm cell sẽ làm test kiểm tra sai thứ.
    """
    if ns is None:
        ns = {}
    cells = _code_cells(nb_path)
    for marker in contains:
        hits = [src for src in cells if marker in src]
        if len(hits) != 1:
            raise LookupError(
                f"marker {marker!r} khop {len(hits)} cell, can dung 1"
            )
        exec(compile(hits[0], f"<nb:{marker}>", "exec"), ns)
    return ns
```

- [ ] **Step 8: Thêm `tests/conftest.py` để pytest thấy được `nbload`**

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 9: Chạy test để xác nhận nó pass**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/ -v
```

Expected: 2 passed.

- [ ] **Step 10: Commit**

```bash
git add .gitignore requirements-dev.txt tests/
git commit -m "test: add notebook-cell loader so notebook code can be unit tested"
```

---

### Task 2: Cell MT helpers

**Files:**
- Modify: `Plant_Disease_UDA_Final.ipynb` (chèn 2 cell trước markdown `## 9 — Results Comparison`)
- Test: `tests/test_mt_helpers.py`

**Interfaces:**
- Consumes: `FeatureExtractor`, `Classifier` (cell 32 của notebook)
- Produces:
  - `sigmoid_rampup(x: float, length: float) -> float`
  - `consistency_weight(epoch: int, cons_max=CONS_MAX, start=CONS_START, ramp_len=CONS_RAMP_LEN) -> float`
  - `softmax_mse_loss(input_logits: Tensor, target_logits: Tensor) -> Tensor` (scalar)
  - `create_teacher(student_phi, student_G, num_classes: int, device) -> tuple[FeatureExtractor, Classifier]`
  - `update_ema(student_mods: list[nn.Module], teacher_mods: list[nn.Module], global_step: int, ema_decay=EMA_DECAY) -> float` (trả về alpha đã dùng)
  - `cdan_entropy_weight(probs: Tensor) -> Tensor` (bản sao có chủ ý của định nghĩa trong cell 42, để section 8.8 không phải chạy cell 42)
  - Hằng số: `EMA_DECAY=0.99`, `CONS_MAX=9.0`, `CONS_START=100`, `CONS_RAMP_LEN=50`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/test_mt_helpers.py`:

```python
import copy
import math

import pytest
import torch
import torch.nn as nn

from nbload import load_cells, NB_PATH

PRELUDE = (
    "import os, copy\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "import torchvision.models as tv_models\n"
)


@pytest.fixture(scope="module")
def ns():
    namespace = {}
    exec(PRELUDE, namespace)
    load_cells(
        NB_PATH,
        contains=["class FeatureExtractor", "CELL-TAG: mt_helpers"],
        ns=namespace,
    )
    return namespace


# ── sigmoid_rampup ────────────────────────────────────────────────────────────

def test_rampup_starts_near_zero_and_saturates_at_one(ns):
    f = ns["sigmoid_rampup"]
    assert f(0, 50) == pytest.approx(math.exp(-5.0))
    assert f(50, 50) == pytest.approx(1.0)


def test_rampup_clips_beyond_length(ns):
    f = ns["sigmoid_rampup"]
    assert f(500, 50) == pytest.approx(1.0)


def test_rampup_is_monotonic(ns):
    f = ns["sigmoid_rampup"]
    vals = [f(x, 50) for x in range(0, 51, 5)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_rampup_zero_length_returns_one(ns):
    assert ns["sigmoid_rampup"](0, 0) == 1.0


# ── consistency_weight ────────────────────────────────────────────────────────

def test_consistency_is_exactly_zero_before_start(ns):
    w = ns["consistency_weight"]
    assert w(1) == 0.0
    assert w(99) == 0.0


def test_consistency_saturates_at_cons_max(ns):
    w = ns["consistency_weight"]
    assert w(150) == pytest.approx(9.0)
    assert w(300) == pytest.approx(9.0)


def test_consistency_midpoint_is_between(ns):
    w = ns["consistency_weight"]
    assert 0.0 < w(125) < 9.0


# ── softmax_mse_loss ──────────────────────────────────────────────────────────

def test_mse_zero_for_identical_logits(ns):
    a = torch.randn(4, 3)
    assert ns["softmax_mse_loss"](a, a).item() == pytest.approx(0.0, abs=1e-10)


def test_mse_positive_for_different_logits(ns):
    a = torch.zeros(4, 3)
    b = torch.tensor([[10.0, 0.0, 0.0]] * 4)
    assert ns["softmax_mse_loss"](a, b).item() > 0.0


def test_mse_divides_by_num_classes(ns):
    """Công thức repo gốc: sum các bình phương sai khác, chia num_classes."""
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[100.0, -100.0]])   # softmax ~ [1, 0]
    expected = ((0.5 - 1.0) ** 2 + (0.5 - 0.0) ** 2) / 2
    assert ns["softmax_mse_loss"](a, b).item() == pytest.approx(expected, abs=1e-6)


# ── update_ema ────────────────────────────────────────────────────────────────

def _two_linears():
    s = nn.Linear(4, 3)
    t = copy.deepcopy(s)
    with torch.no_grad():
        s.weight.fill_(1.0)
        t.weight.fill_(0.0)
    return s, t


def test_ema_alpha_is_zero_at_step_zero(ns):
    """alpha = min(1 - 1/(0+1), 0.99) = 0 → teacher bị ghi đè thành student."""
    s, t = _two_linears()
    alpha = ns["update_ema"]([s], [t], global_step=0)
    assert alpha == 0.0
    assert torch.allclose(t.weight, torch.ones_like(t.weight))


def test_ema_alpha_is_half_at_step_one(ns):
    s, t = _two_linears()
    alpha = ns["update_ema"]([s], [t], global_step=1)
    assert alpha == pytest.approx(0.5)
    assert torch.allclose(t.weight, torch.full_like(t.weight, 0.5))


def test_ema_alpha_caps_at_decay(ns):
    s, t = _two_linears()
    alpha = ns["update_ema"]([s], [t], global_step=10_000, ema_decay=0.99)
    assert alpha == pytest.approx(0.99)
    assert torch.allclose(t.weight, torch.full_like(t.weight, 0.01))


def test_ema_does_not_touch_buffers(ns):
    """BN buffer KHÔNG được EMA — đó là lý do teacher phải ở train() mode."""
    s = nn.BatchNorm1d(4)
    t = copy.deepcopy(s)
    with torch.no_grad():
        s.running_mean.fill_(5.0)
    ns["update_ema"]([s], [t], global_step=1)
    assert torch.allclose(t.running_mean, torch.zeros(4))


# ── create_teacher ────────────────────────────────────────────────────────────

def test_teacher_starts_identical_to_student(ns):
    phi = ns["FeatureExtractor"](pretrained=False)
    G = ns["Classifier"](phi.out_dim, 3)
    phi_t, G_t = ns["create_teacher"](phi, G, 3, "cpu")
    assert torch.allclose(G_t.fc.weight, G.fc.weight)


def test_teacher_params_require_no_grad(ns):
    phi = ns["FeatureExtractor"](pretrained=False)
    G = ns["Classifier"](phi.out_dim, 3)
    phi_t, G_t = ns["create_teacher"](phi, G, 3, "cpu")
    assert all(not p.requires_grad for p in phi_t.parameters())
    assert all(not p.requires_grad for p in G_t.parameters())


def test_teacher_output_carries_no_grad(ns):
    phi = ns["FeatureExtractor"](pretrained=False)
    G = ns["Classifier"](phi.out_dim, 3)
    phi_t, G_t = ns["create_teacher"](phi, G, 3, "cpu")
    phi_t.eval(); G_t.eval()
    out = G_t(phi_t(torch.randn(2, 3, 224, 224)))
    assert out.requires_grad is False


# ── Hành vi BatchNorm — cạm bẫy chính của thiết kế ───────────────────────────

def test_bn_buffers_move_in_train_mode(ns):
    """Teacher ở train() thì BN running stats phải tự cập nhật."""
    phi = ns["FeatureExtractor"](pretrained=False)
    G = ns["Classifier"](phi.out_dim, 3)
    phi_t, _ = ns["create_teacher"](phi, G, 3, "cpu")
    phi_t.train()
    before = phi_t.net[1].running_mean.clone()
    with torch.no_grad():
        phi_t(torch.randn(4, 3, 224, 224))
    assert not torch.allclose(before, phi_t.net[1].running_mean)


def test_bn_buffers_frozen_in_eval_mode(ns):
    phi = ns["FeatureExtractor"](pretrained=False)
    G = ns["Classifier"](phi.out_dim, 3)
    phi_t, _ = ns["create_teacher"](phi, G, 3, "cpu")
    phi_t.eval()
    before = phi_t.net[1].running_mean.clone()
    with torch.no_grad():
        phi_t(torch.randn(4, 3, 224, 224))
    assert torch.allclose(before, phi_t.net[1].running_mean)
```

- [ ] **Step 2: Chạy test để xác nhận nó thất bại**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/test_mt_helpers.py -v
```

Expected: tất cả FAIL ở fixture với `LookupError: marker 'CELL-TAG: mt_helpers' khop 0 cell, can dung 1`.

- [ ] **Step 3: Chèn cell markdown mở đầu section 8.8**

Chèn ngay **trước** cell markdown chứa `## 9 — Results Comparison`. Loại cell: `markdown`.

```markdown
## 8.8 — CDAN+E + Mean Teacher

Kết hợp CDAN+E (alignment đối kháng có điều kiện theo lớp) với Mean Teacher
(consistency regularization trên target không nhãn), theo tinh thần
*French et al., "Self-Ensembling for Visual Domain Adaptation", ICLR 2018*.

```
L = L_cls + L_cdan + w(t) · L_cons
```

Hai chi tiết lấy từ source gốc, không suy ra được từ paper:

1. `losses.py` của `milsever/plant-pathology` **không** detach teacher logits — repo
   chặn gradient bằng `freeze_all()` lúc khởi tạo. Ở đây ta làm cả hai:
   `requires_grad_(False)` và `torch.no_grad()`.
2. `update_ema_variables` của `CuriousAI/mean-teacher` chỉ duyệt `.parameters()`,
   **không** đụng buffer BatchNorm. Official bù lại bằng cách để teacher ở
   `.train()` mode suốt quá trình huấn luyện, nên BN running stats tự cập nhật.
   ResNet-18 có 20 lớp BN — để nhầm `.eval()` là teacher hỏng hoàn toàn.

`w(t)` ramp **trễ** (epoch 100 → 150) chứ không ramp sớm như paper SSL, vì ở đây
λ_GRL cũng đang ramp; ép consistency lúc feature còn xáo trộn mạnh dễ gây
confirmation bias.
```

- [ ] **Step 4: Chèn cell code MT helpers**

Chèn ngay sau cell markdown vừa tạo. Loại cell: `code`.

```python
# === CELL-TAG: mt_helpers ===
# ══════════════════════════════════════════════════════════════════════════════
# MEAN TEACHER — hằng số, lịch ramp, loss, EMA
# Nguồn: github.com/CuriousAI/mean-teacher và github.com/milsever/plant-pathology
# ══════════════════════════════════════════════════════════════════════════════

EMA_DECAY     = 0.99   # paper Ilsever & Baz, lấy từ test "4000 labels/ResNet CIFAR-10"
CONS_MAX      = 9.0    # = C^2; paper chọn consistency weight trong [C, C^2], C=3
CONS_START    = 100    # epoch bắt đầu ramp (sau khi lambda_GRL đã gần bão hoà)
CONS_RAMP_LEN = 50     # số epoch để ramp tới CONS_MAX


def sigmoid_rampup(x, length):
    """Sigmoid ramp-up của Laine & Aila (ramps.py gốc).

    Args:
        x     : Số epoch đã trôi qua kể từ lúc bắt đầu ramp.
        length: Số epoch để đạt bão hoà. 0 nghĩa là không ramp.
    Returns:
        Hệ số trong (0, 1]. Tại x=0 trả về exp(-5) ~ 0.0067, không phải 0.
    """
    if length == 0:
        return 1.0
    x = float(np.clip(x, 0.0, length)) / length
    return float(np.exp(-5.0 * (1.0 - x) ** 2))


def consistency_weight(epoch, cons_max=CONS_MAX, start=CONS_START,
                       ramp_len=CONS_RAMP_LEN):
    """Trọng số w(t) của consistency loss. Đúng 0 trước `start`."""
    if epoch < start:
        return 0.0
    return cons_max * sigmoid_rampup(epoch - start, ramp_len)


def softmax_mse_loss(input_logits, target_logits):
    """MSE giữa hai phân phối softmax. Sao chép nguyên văn losses.py của repo gốc.

    Trả về TỔNG trên toàn batch (chia num_classes). Chỗ gọi phải chia batch_size,
    đúng như ft_meanteacher.py làm.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax  = F.softmax(input_logits,  dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes    = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def cdan_entropy_weight(probs):
    """Trọng số entropy của CDAN+E: w = 1 + exp(-H(p)).

    Sao chép nguyên văn định nghĩa trong cell 42. Cố ý lặp lại ở đây để section
    8.8 chạy được mà KHÔNG phải chạy cell 42 — cell đó là một run 300 epoch đầy
    đủ. Hai định nghĩa giống hệt nhau; cell 8.8 chạy sau nên định nghĩa này
    thắng, và kết quả không đổi dù chạy theo thứ tự nào.

    Tin cậy (H thấp) -> ~2, mơ hồ -> ~1. Detach để làm hệ số, không phải đường
    truyền gradient.
    """
    ent = -(probs * torch.log(probs + 1e-5)).sum(dim=1)
    return (1.0 + torch.exp(-ent)).detach()


def create_teacher(student_phi, student_G, num_classes, device):
    """Dựng teacher khởi tạo giống hệt student và đã khoá gradient.

    pretrained=False để khỏi tải lại weight ImageNet — trọng số bị ghi đè ngay
    bằng state_dict của student.

    Returns:
        (phi_t, G_t) — cả hai đã requires_grad_(False).
    """
    phi_t = FeatureExtractor(pretrained=False).to(device)
    G_t   = Classifier(phi_t.out_dim, num_classes).to(device)
    phi_t.load_state_dict(student_phi.state_dict())
    G_t.load_state_dict(student_G.state_dict())
    for p in list(phi_t.parameters()) + list(G_t.parameters()):
        p.requires_grad_(False)
    return phi_t, G_t


def update_ema(student_mods, teacher_mods, global_step, ema_decay=EMA_DECAY):
    """Cập nhật EMA. Sao chép update_ema_variables của CuriousAI/mean-teacher.

    Dùng true average cho tới khi exponential average đủ chính xác, nên alpha
    tăng dần từ 0 lên `ema_decay`.

    LƯU Ý: chỉ duyệt .parameters(), KHÔNG đụng buffer BatchNorm. Buffer của
    teacher được cập nhật nhờ chính forward pass của nó ở .train() mode.

    Returns:
        alpha đã dùng (để log).
    """
    alpha = min(1 - 1 / (global_step + 1), ema_decay)
    with torch.no_grad():
        for sm, tm in zip(student_mods, teacher_mods):
            for ema_p, p in zip(tm.parameters(), sm.parameters()):
                ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)
    return alpha


print('Mean Teacher helpers defined.')
```

- [ ] **Step 5: Chạy test để xác nhận nó pass**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/test_mt_helpers.py -v
```

Expected: 19 passed.

- [ ] **Step 6: Commit**

```bash
git add Plant_Disease_UDA_Final.ipynb tests/test_mt_helpers.py
git commit -m "feat: add Mean Teacher helpers (rampup, softmax MSE, EMA) with tests"
```

---

### Task 3: Loader target hai view

**Files:**
- Modify: `Plant_Disease_UDA_Final.ipynb` (chèn 1 cell sau cell `mt_helpers`)
- Test: `tests/test_two_view.py`

**Interfaces:**
- Consumes: `PlantDiseaseDataset`, `TRAIN_TRANSFORM`, `CLASS_NAMES`, `make_loader`, `dataset_B_target` (các cell sẵn có)
- Produces:
  - `class TwoCropTransform` — `__call__(img) -> tuple[Tensor, Tensor]`
  - `dataset_B_target_2view: PlantDiseaseDataset`
  - `target_loader_2view: DataLoader` — mỗi batch là `((v1, v2), labels)` với `v1.shape == v2.shape == (64, 3, 224, 224)` và `labels` toàn `-1`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/test_two_view.py`:

```python
import pytest
import torch
from PIL import Image

from nbload import load_cells, NB_PATH

PRELUDE = (
    "import os, copy, random\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "import torchvision.transforms as T\n"
    "from torch.utils.data import Dataset, DataLoader, Subset\n"
    "from PIL import Image\n"
)


@pytest.fixture(scope="module")
def ns():
    namespace = {}
    exec(PRELUDE, namespace)
    load_cells(
        NB_PATH,
        contains=["TRAIN_TRANSFORM = T.Compose", "CELL-TAG: two_view"],
        ns=namespace,
    )
    return namespace


def test_two_views_have_correct_shape(ns):
    img = Image.new("RGB", (300, 300), (120, 160, 90))
    v1, v2 = ns["TwoCropTransform"](ns["TRAIN_TRANSFORM"])(img)
    assert v1.shape == (3, 224, 224)
    assert v2.shape == (3, 224, 224)


def test_two_views_actually_differ(ns):
    """Nếu hai view giống nhau thì L_cons luôn = 0 và Mean Teacher vô tác dụng.

    Dùng ảnh có cấu trúc (gradient màu) để RandomCrop/Flip tạo khác biệt thật;
    ảnh phẳng một màu sẽ cho hai view giống nhau bất kể crop.
    """
    torch.manual_seed(0)
    arr = torch.arange(300 * 300 * 3, dtype=torch.uint8).reshape(300, 300, 3)
    img = Image.fromarray(arr.numpy(), mode="RGB")
    v1, v2 = ns["TwoCropTransform"](ns["TRAIN_TRANSFORM"])(img)
    assert not torch.allclose(v1, v2)


def test_val_transform_would_produce_identical_views(ns):
    """Chốt lại lý do phải dùng TRAIN_TRANSFORM: VAL_TRANSFORM là tất định.

    Đây chính là cái bẫy — dataset_B_target gốc dùng VAL_TRANSFORM.
    """
    arr = torch.arange(300 * 300 * 3, dtype=torch.uint8).reshape(300, 300, 3)
    img = Image.fromarray(arr.numpy(), mode="RGB")
    v1, v2 = ns["TwoCropTransform"](ns["VAL_TRANSFORM"])(img)
    assert torch.allclose(v1, v2)
```

- [ ] **Step 2: Chạy test để xác nhận nó thất bại**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/test_two_view.py -v
```

Expected: FAIL với `LookupError: marker 'CELL-TAG: two_view' khop 0 cell, can dung 1`.

- [ ] **Step 3: Chèn cell code `TwoCropTransform`**

Chèn ngay sau cell `mt_helpers`. Loại cell: `code`.

Cell này **chỉ chứa định nghĩa class**, không đụng tới dữ liệu. Đó là điều kiện để test cục bộ exec được nó mà không cần Colab hay dataset — phần dựng loader nằm ở cell kế tiếp, không mang tag.

```python
# === CELL-TAG: two_view ===
# ══════════════════════════════════════════════════════════════════════════════
# TWO-CROP TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

class TwoCropTransform:
    """Sinh 2 view độc lập của cùng một ảnh.

    Nguồn ngẫu nhiên duy nhất là 2 lần bốc mẫu khác nhau của RandomCrop /
    RandomHorizontalFlip / RandomVerticalFlip / ColorJitter bên trong transform.
    Cả hai view đều là 'weak' — Ilsever & Baz thử cả weak và strong trên chính
    dataset lá cây này và weak cho kết quả tốt hơn.

    CẢNH BÁO: transform truyền vào PHẢI có thành phần ngẫu nhiên. Bọc quanh
    VAL_TRANSFORM (tất định) sẽ cho hai view giống hệt nhau, L_cons luôn bằng 0,
    và Mean Teacher không làm gì cả suốt 300 epoch mà không báo lỗi nào.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


print('TwoCropTransform defined.')
```

- [ ] **Step 4: Chèn cell dựng loader hai view**

Chèn ngay sau cell `two_view`. Loại cell: `code`. Cell này **không** mang CELL-TAG vì nó cần dữ liệu thật, chỉ chạy được trên Colab.

```python
# ══════════════════════════════════════════════════════════════════════════════
# LOADER TARGET HAI VIEW
#
# CẢNH BÁO: dataset_B_target gốc (cell 13, split_ppd) được tạo với VAL_TRANSFORM
# — chỉ Resize, hoàn toàn tất định. Nếu bọc TwoCropTransform quanh nó thì hai
# view sẽ giống hệt nhau từng pixel và Mean Teacher trở thành no-op.
#
# Nên ở đây ta dựng lại dataset từ cùng danh sách samples, nhưng với
# TRAIN_TRANSFORM (có RandomCrop/Flip/ColorJitter) và labeled=False.
#
# CẢ HAI nhánh (baseline và MT) đều dùng loader này — nhánh baseline chỉ lấy
# view 2 và bỏ view 1. Nhờ vậy hai nhánh tiêu thụ RNG y hệt nhau, nên so sánh
# ghép cặp theo seed mới là thật, và biến độc lập duy nhất đúng là L_cons.
# ══════════════════════════════════════════════════════════════════════════════

# dataset_B_target là một Subset; lấy lại danh sách (path, label) gốc của nó.
_target_samples = [
    dataset_B_target.dataset.samples[i] for i in dataset_B_target.indices
]

dataset_B_target_2view = PlantDiseaseDataset(
    samples   = _target_samples,
    classes   = CLASS_NAMES,
    transform = TwoCropTransform(TRAIN_TRANSFORM),
    labeled   = False,          # trả về -1 thay vì nhãn thật: chặn rò rỉ nhãn target
)

target_loader_2view = make_loader(
    dataset_B_target_2view, batch_size=64, shuffle=True, drop_last=True
)

# ── Kiểm tra tại chỗ: hai view phải KHÁC nhau ────────────────────────────────
(_v1, _v2), _lbl = next(iter(target_loader_2view))
assert _v1.shape == _v2.shape == (64, 3, 224, 224), f'sai shape: {_v1.shape}'
assert not torch.allclose(_v1, _v2), \
    'Hai view giong het nhau -> L_cons se luon = 0. Kiem tra transform.'
assert (_lbl == -1).all(), 'Target loader dang tra ve nhan that.'

print(f'Two-view target loader ready: {len(dataset_B_target_2view)} anh')
print(f'  view khac nhau: mean|v1-v2| = {(_v1 - _v2).abs().mean():.4f}')
del _v1, _v2, _lbl
```

- [ ] **Step 5: Chạy test để xác nhận nó pass**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/test_two_view.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Chạy trên Colab**

Mở notebook trên Colab, chạy lại từ đầu tới hết cell dựng loader.

Expected output:
```
Two-view target loader ready: 1038 anh
  view khac nhau: mean|v1-v2| = 0.xxxx
```
Không có AssertionError. Nếu `mean|v1-v2|` bằng 0.0000 thì transform sai — dừng lại, đừng chạy tiếp.

- [ ] **Step 7: Commit**

```bash
git add Plant_Disease_UDA_Final.ipynb tests/test_two_view.py
git commit -m "feat: add two-view target loader with augmentation (fixes silent zero consistency loss)"
```

---

### Task 4: Checkpoint có state của teacher

**Files:**
- Modify: `Plant_Disease_UDA_Final.ipynb` (chèn 1 cell sau cell `two_view`)
- Test: `tests/test_mt_checkpoint.py`

**Interfaces:**
- Consumes: không (độc lập với `save_uda_checkpoint` của cell 34 — không sửa cell đó)
- Produces:
  - `save_mt_checkpoint(path, epoch, step, best_loss, best_state, phi, G, D, opt, sch, phi_t=None, G_t=None) -> None`
  - `load_mt_checkpoint(path, phi, G, D, opt, sch, phi_t=None, G_t=None, device='cpu') -> tuple[int, int, float, dict | None]` trả về `(start_epoch, step, best_loss, best_state)`; nếu file không tồn tại trả `(1, 0, inf, None)`

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/test_mt_checkpoint.py`:

```python
import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from nbload import load_cells, NB_PATH

PRELUDE = (
    "import os, copy\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.optim as optim\n"
    "import torch.nn.functional as F\n"
    "import torchvision.models as tv_models\n"
)


@pytest.fixture(scope="module")
def ns():
    namespace = {}
    exec(PRELUDE, namespace)
    load_cells(NB_PATH, contains=["CELL-TAG: mt_checkpoint"], ns=namespace)
    return namespace


def _bundle():
    phi = nn.Linear(4, 4)
    G = nn.Linear(4, 3)
    D = nn.Linear(12, 2)
    phi_t = nn.Linear(4, 4)
    G_t = nn.Linear(4, 3)
    opt = optim.AdamW(list(phi.parameters()) + list(G.parameters()) + list(D.parameters()))
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    return phi, G, D, phi_t, G_t, opt, sch


def test_returns_fresh_start_when_file_missing(ns, tmp_path):
    phi, G, D, phi_t, G_t, opt, sch = _bundle()
    out = ns["load_mt_checkpoint"](
        str(tmp_path / "nope.pth"), phi, G, D, opt, sch, phi_t, G_t
    )
    assert out[0] == 1
    assert out[1] == 0
    assert math.isinf(out[2])
    assert out[3] is None


def test_roundtrip_restores_teacher_weights(ns, tmp_path):
    phi, G, D, phi_t, G_t, opt, sch = _bundle()
    with torch.no_grad():
        G_t.weight.fill_(7.0)
    p = str(tmp_path / "ck.pth")
    ns["save_mt_checkpoint"](p, 42, 1234, 0.5, {"phi": phi.state_dict()},
                             phi, G, D, opt, sch, phi_t, G_t)

    phi2, G2, D2, phi_t2, G_t2, opt2, sch2 = _bundle()
    start_epoch, step, best_loss, best_state = ns["load_mt_checkpoint"](
        p, phi2, G2, D2, opt2, sch2, phi_t2, G_t2
    )
    assert start_epoch == 43          # resume ở epoch KẾ TIẾP
    assert step == 1234
    assert best_loss == pytest.approx(0.5)
    assert best_state is not None
    assert torch.allclose(G_t2.weight, torch.full_like(G_t2.weight, 7.0))


def test_roundtrip_works_without_teacher(ns, tmp_path):
    """Nhánh baseline (use_mt=False) không có teacher."""
    phi, G, D, _, _, opt, sch = _bundle()
    p = str(tmp_path / "ck.pth")
    ns["save_mt_checkpoint"](p, 5, 10, 1.0, None, phi, G, D, opt, sch, None, None)

    phi2, G2, D2, _, _, opt2, sch2 = _bundle()
    start_epoch, step, best_loss, best_state = ns["load_mt_checkpoint"](
        p, phi2, G2, D2, opt2, sch2, None, None
    )
    assert start_epoch == 6
    assert best_state is None
```

- [ ] **Step 2: Chạy test để xác nhận nó thất bại**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/test_mt_checkpoint.py -v
```

Expected: FAIL với `LookupError: marker 'CELL-TAG: mt_checkpoint' khop 0 cell, can dung 1`.

- [ ] **Step 3: Chèn cell code checkpoint**

Chèn ngay sau cell dựng loader hai view. Loại cell: `code`.

```python
# === CELL-TAG: mt_checkpoint ===
# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT cho nhánh CDAN+E (+MT)
#
# Không sửa save_uda_checkpoint/load_uda_checkpoint của cell 34 — chúng đang
# phục vụ 5 method cũ. Đây là cặp riêng, có thêm state của teacher.
#
# 10 run x 300 epoch trên Colab chắc chắn sẽ bị ngắt session giữa chừng, nên
# resume là bắt buộc chứ không phải tuỳ chọn.
# ══════════════════════════════════════════════════════════════════════════════

def save_mt_checkpoint(path, epoch, step, best_loss, best_state,
                       phi, G, D, opt, sch, phi_t=None, G_t=None):
    """Lưu toàn bộ state để resume được.

    Args:
        epoch     : Epoch VỪA HOÀN THÀNH.
        step      : Bộ đếm iteration toàn cục (giữ liên tục lịch lambda và EMA).
        best_state: dict {'phi','G'} của model tốt nhất, hoặc None nếu chưa có.
        phi_t, G_t: Teacher; None với nhánh baseline.
    """
    ckpt = {
        'epoch': epoch, 'step': step,
        'best_loss': best_loss, 'best_state': best_state,
        'phi': phi.state_dict(), 'G': G.state_dict(), 'D': D.state_dict(),
        'optimizer': opt.state_dict(), 'scheduler': sch.state_dict(),
    }
    if phi_t is not None:
        # state_dict của teacher gồm cả BUFFER BatchNorm. Bắt buộc phải lưu:
        # chúng không được EMA mà tích luỹ dần qua forward pass, nên mất là
        # teacher quay về BN stats của ImageNet.
        ckpt['phi_t'] = phi_t.state_dict()
        ckpt['G_t']   = G_t.state_dict()
    torch.save(ckpt, path)


def load_mt_checkpoint(path, phi, G, D, opt, sch,
                       phi_t=None, G_t=None, device='cpu'):
    """Khôi phục state nếu checkpoint tồn tại.

    Returns:
        (start_epoch, step, best_loss, best_state).
        Nếu không có file: (1, 0, inf, None).
    """
    if not os.path.exists(path):
        return 1, 0, float('inf'), None

    ckpt = torch.load(path, map_location=device)
    phi.load_state_dict(ckpt['phi'])
    G.load_state_dict(ckpt['G'])
    D.load_state_dict(ckpt['D'])
    opt.load_state_dict(ckpt['optimizer'])
    sch.load_state_dict(ckpt['scheduler'])
    if phi_t is not None and 'phi_t' in ckpt:
        phi_t.load_state_dict(ckpt['phi_t'])
        G_t.load_state_dict(ckpt['G_t'])
    return ckpt['epoch'] + 1, ckpt['step'], ckpt['best_loss'], ckpt['best_state']


print('MT checkpoint helpers defined.')
```

- [ ] **Step 4: Chạy test để xác nhận nó pass**

```bash
cd "/Users/tanvo/Documents/GRAD/THESIS/RELATED REPO/MeanTeacher_CDANE"
.venv/bin/python -m pytest tests/ -v
```

Expected: 27 passed (2 nbload + 19 mt_helpers + 3 two_view + 3 mt_checkpoint).

- [ ] **Step 5: Commit**

```bash
git add Plant_Disease_UDA_Final.ipynb tests/test_mt_checkpoint.py
git commit -m "feat: add MT checkpoint helpers that persist teacher BN buffers"
```

---

### Task 5: Hàm huấn luyện hợp nhất

**Files:**
- Modify: `Plant_Disease_UDA_Final.ipynb` (chèn 1 cell sau cell `mt_checkpoint`)

**Interfaces:**
- Consumes: `FeatureExtractor`, `Classifier`, `CDANDisc`, `GradRevLayer`, `multilinear_map`, `get_lambda`, `domain_lbl`, `uda_test` (cell 32); `set_seed`, `DEVICE`, `NUM_CLASSES`, `SAVE_DIR` (cell 2/4); `train_loader_A_fbr`, `val_loader_A_fbr`, `test_loader_B` (cell 14/27); `create_teacher`, `update_ema`, `consistency_weight`, `softmax_mse_loss`, `cdan_entropy_weight` (Task 2); `target_loader_2view` (Task 3); `save_mt_checkpoint`, `load_mt_checkpoint` (Task 4)
- **Không** phụ thuộc cell 42 — `cdan_entropy_weight` đã được định nghĩa lại ở Task 2 đúng vì lý do này. Section 8.8 chạy được mà không cần chạy 300 epoch của cell 42.
- Produces: `train_cdane(seed: int, use_mt: bool, max_epochs=300, min_sel=250, ckpt_dir=None, log_every=10, cons_max=CONS_MAX, cons_start=CONS_START, cons_ramp_len=CONS_RAMP_LEN) -> dict` với các khoá `seed`, `use_mt`, `acc_selected`, `acc_student_final`, `best_val_loss`, `epochs_run`

**Chú ý về TDD ở task này:** hàm này cần GPU và dữ liệu thật nên không unit-test cục bộ được. Cơ chế kiểm chứng là **smoke run 4 epoch trên Colab với ngưỡng ramp hạ thấp**, chạy trước khi commit bất kỳ run dài nào. Đó là Step 2 và 3.

- [ ] **Step 1: Chèn cell code hàm huấn luyện**

Chèn ngay sau cell `mt_checkpoint`. Loại cell: `code`.

```python
# === CELL-TAG: train_cdane ===
# ══════════════════════════════════════════════════════════════════════════════
# HÀM HUẤN LUYỆN HỢP NHẤT — phục vụ CẢ hai nhánh
#
# Chỉ có MỘT đường code cho cả baseline lẫn MT. `use_mt` là biến độc lập duy
# nhất. Viết hai hàm riêng sẽ mở đường cho những khác biệt vô tình (thứ tự RNG,
# tiêu chí chọn model, cách augment) len vào và làm hỏng phép so sánh.
#
# Teacher được tạo trong CẢ hai nhánh, kể cả khi use_mt=False. Nghe có vẻ thừa,
# nhưng nn.Linear khởi tạo có tiêu thụ RNG — không tạo teacher ở nhánh baseline
# sẽ làm hai nhánh lệch dòng RNG và phá tính ghép cặp theo seed.
# ══════════════════════════════════════════════════════════════════════════════

def train_cdane(seed, use_mt, max_epochs=300, min_sel=250, ckpt_dir=None,
                log_every=10, cons_max=CONS_MAX, cons_start=CONS_START,
                cons_ramp_len=CONS_RAMP_LEN, verbose=True):
    """Huấn luyện CDAN+E, có hoặc không có Mean Teacher.

    Args:
        seed         : Seed cho set_seed(). Cùng seed ở hai nhánh -> ghép cặp được.
        use_mt       : True thì cộng thêm w(t)*L_cons và chọn model theo teacher.
        max_epochs   : Tổng số epoch (paper: 300 cho adversarial UDA).
        min_sel      : Chỉ bắt đầu chọn model sau epoch này (paper: 250).
        ckpt_dir     : Thư mục lưu checkpoint. None thì không lưu (dùng cho smoke test).
        cons_start   : Epoch bắt đầu ramp consistency.
        cons_ramp_len: Số epoch để ramp tới cons_max.

    Returns:
        dict: seed, use_mt, acc_selected (%), acc_student_final (%),
              best_val_loss, epochs_run.
    """
    tag = 'mt' if use_mt else 'base'
    set_seed(seed)

    # ── Khởi tạo (thứ tự này cố định để dòng RNG giống nhau ở hai nhánh) ──────
    phi = FeatureExtractor(pretrained=True).to(DEVICE)
    G   = Classifier(phi.out_dim, NUM_CLASSES).to(DEVICE)
    D   = CDANDisc(phi.out_dim, NUM_CLASSES).to(DEVICE)
    grl = GradRevLayer().to(DEVICE)
    phi_t, G_t = create_teacher(phi, G, NUM_CLASSES, DEVICE)   # tạo ở CẢ hai nhánh

    opt = optim.AdamW(
        list(phi.parameters()) + list(G.parameters()) + list(D.parameters()),
        lr=1e-3, weight_decay=0.01
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    ce_mean   = nn.CrossEntropyLoss()
    ce_persam = nn.CrossEntropyLoss(reduction='none')

    ckpt_path = best_path = None
    start_epoch, step, best_loss, best_state = 1, 0, float('inf'), None
    if ckpt_dir is not None:
        ckpt_path = os.path.join(ckpt_dir, f'cdane_{tag}_s{seed}_ckpt.pth')
        best_path = os.path.join(ckpt_dir, f'cdane_{tag}_s{seed}_best.pth')
        start_epoch, step, best_loss, best_state = load_mt_checkpoint(
            ckpt_path, phi, G, D, opt, sch, phi_t, G_t, device=DEVICE
        )
        if start_epoch > 1 and verbose:
            print(f'  [resume] tiep tuc tu epoch {start_epoch}')

    steps_per_epoch = min(len(train_loader_A_fbr), len(target_loader_2view))
    total_steps     = max_epochs * steps_per_epoch

    for epoch in range(start_epoch, max_epochs + 1):
        phi.train(); G.train(); D.train()
        phi_t.train(); G_t.train()      # BẮT BUỘC train(): BN buffer của teacher
                                        # không được EMA, chỉ cập nhật qua forward
        w_cons = consistency_weight(epoch, cons_max, cons_start, cons_ramp_len)

        ep_cls = ep_cdan = ep_cons = 0.0
        tgt_hist = torch.zeros(NUM_CLASSES, dtype=torch.long)

        for (si, sl), ((tv1, tv2), _) in zip(train_loader_A_fbr, target_loader_2view):
            si, sl   = si.to(DEVICE), sl.to(DEVICE)
            tv1, tv2 = tv1.to(DEVICE), tv2.to(DEVICE)

            lam = get_lambda(step, total_steps)
            grl.set_lam(lam)
            step += 1
            opt.zero_grad()

            # ── Nhánh CDAN+E: y hệt cell 42, target dùng view 2 ──────────────
            sf = phi(si)
            tf = phi(tv2)
            logits_s = G(sf)
            logits_t = G(tf)
            sp = torch.softmax(logits_s, dim=1)
            tp = torch.softmax(logits_t, dim=1)

            L_cls = ce_mean(logits_s, sl)

            # Detach PREDICTION (đúng CDAN gốc) nhưng giữ gradient của FEATURE,
            # để tín hiệu đối kháng còn tới được phi qua GRL.
            src_mm = grl(multilinear_map(sf, sp.detach()))
            tgt_mm = grl(multilinear_map(tf, tp.detach()))

            sw = cdan_entropy_weight(sp); sw = sw / sw.sum()
            tw = cdan_entropy_weight(tp); tw = tw / tw.sum()
            src_d = (sw * ce_persam(D(src_mm), domain_lbl(si.size(0),  True,  DEVICE))).sum()
            tgt_d = (tw * ce_persam(D(tgt_mm), domain_lbl(tv2.size(0), False, DEVICE))).sum()
            L_cdan = (src_d + tgt_d) / 2.0

            # ── Teacher forward: luôn chạy, để có chẩn đoán ngay từ epoch 1 ──
            with torch.no_grad():
                teacher_logits = G_t(phi_t(tv1))
            tgt_hist += torch.bincount(
                teacher_logits.argmax(1).cpu(), minlength=NUM_CLASSES
            )

            # ── Consistency: student nhìn view 2 (tái dùng logits_t) ─────────
            L_cons = softmax_mse_loss(logits_t, teacher_logits) / tv2.size(0)

            total = L_cls + L_cdan + (w_cons * L_cons if use_mt else 0.0)
            total.backward()
            opt.step()

            if use_mt:
                update_ema([phi, G], [phi_t, G_t], step)

            ep_cls  += L_cls.item()
            ep_cdan += L_cdan.item()
            ep_cons += L_cons.item()

        sch.step()

        # ── Validation trên FBR val (source domain — hợp lệ với UDA) ─────────
        sel_phi, sel_G = (phi_t, G_t) if use_mt else (phi, G)
        sel_phi.eval(); sel_G.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for vi, vlab in val_loader_A_fbr:
                vi, vlab = vi.to(DEVICE), vlab.to(DEVICE)
                logits = sel_G(sel_phi(vi))
                vl += ce_mean(logits, vlab).item()
                vc += (logits.argmax(1) == vlab).sum().item()
                vt += vlab.size(0)
        val_loss, val_acc = vl / len(val_loader_A_fbr), vc / vt

        if epoch >= min_sel and val_loss < best_loss:
            best_loss  = val_loss
            best_state = {'phi': copy.deepcopy(sel_phi.state_dict()),
                          'G':   copy.deepcopy(sel_G.state_dict())}
            if best_path:
                torch.save(best_state, best_path)

        if verbose and (epoch % log_every == 0 or epoch == max_epochs):
            share = (tgt_hist.float() / max(tgt_hist.sum().item(), 1) * 100)
            print(f'  [{tag} s{seed}] ep {epoch:03d}/{max_epochs} '
                  f'val_acc={val_acc*100:5.1f}% lam={lam:.3f} w={w_cons:.2f} '
                  f'| cls={ep_cls/steps_per_epoch:.3f} '
                  f'cdan={ep_cdan/steps_per_epoch:.3f} '
                  f'cons={ep_cons/steps_per_epoch:.4f} '
                  f'| tgt%=[{share[0]:.0f},{share[1]:.0f},{share[2]:.0f}]')
            if share.max().item() > 70.0:
                print(f'     CANH BAO: teacher gan sap ve 1 lop '
                      f'({share.max():.0f}%) — nghi ngo class collapse')

        if ckpt_path and (epoch % log_every == 0 or epoch == max_epochs):
            save_mt_checkpoint(ckpt_path, epoch, step, best_loss, best_state,
                               phi, G, D, opt, sch, phi_t, G_t)

    # ── Đánh giá ─────────────────────────────────────────────────────────────
    if best_state is None:
        print(f'  CANH BAO [{tag} s{seed}]: chua toi cua so chon model, dung weight cuoi.')
        best_state = {'phi': copy.deepcopy(sel_phi.state_dict()),
                      'G':   copy.deepcopy(sel_G.state_dict())}

    eval_phi = FeatureExtractor(pretrained=False).to(DEVICE)
    eval_G   = Classifier(eval_phi.out_dim, NUM_CLASSES).to(DEVICE)
    eval_phi.load_state_dict(best_state['phi'])
    eval_G.load_state_dict(best_state['G'])

    acc_selected      = uda_test(eval_phi, eval_G, test_loader_B, DEVICE) * 100
    acc_student_final = uda_test(phi, G, test_loader_B, DEVICE) * 100

    if use_mt and acc_student_final > acc_selected + 3.0:
        print(f'  CANH BAO: student ({acc_student_final:.2f}%) hon han teacher '
              f'({acc_selected:.2f}%) — nghi ngo BN buffer chua sync')

    return {
        'seed': seed, 'use_mt': use_mt,
        'acc_selected': acc_selected,
        'acc_student_final': acc_student_final,
        'best_val_loss': best_loss,
        'epochs_run': max_epochs,
    }


print('train_cdane() defined.')
```

- [ ] **Step 2: Chạy smoke test trên Colab**

Chạy trong một cell tạm (không lưu vào notebook). Ép ramp xảy ra trong 4 epoch để đường code MT thực sự được thực thi.

```python
smoke_base = train_cdane(seed=1, use_mt=False, max_epochs=4, min_sel=3,
                         ckpt_dir=None, log_every=1,
                         cons_start=2, cons_ramp_len=2)
print(smoke_base)
smoke_mt   = train_cdane(seed=1, use_mt=True,  max_epochs=4, min_sel=3,
                         ckpt_dir=None, log_every=1,
                         cons_start=2, cons_ramp_len=2)
print(smoke_mt)
```

Expected — kiểm tra từng điểm một:

1. Cả hai chạy hết không lỗi.
2. Log của `mt` cho thấy `w=0.00` ở epoch 1, `w>0` từ epoch 2 trở đi.
3. `cons=` là số dương hữu hạn ở cả hai nhánh (baseline vẫn tính `L_cons` để chẩn đoán, chỉ không cộng vào loss).
4. **`cons` KHÔNG được bằng 0.0000** — nếu bằng 0 thì hai view giống nhau, quay lại Task 3.
5. `tgt%` không phải `[100,0,0]` ngay từ epoch 1.
6. Không có dòng `CANH BAO`.
7. `smoke_base['acc_selected']` khác `smoke_mt['acc_selected']` — nếu bằng nhau chính xác thì `use_mt` chưa có tác dụng gì.

- [ ] **Step 3: Kiểm tra teacher thực sự dịch chuyển khỏi student**

Chạy trong cell tạm:

```python
set_seed(1)
_phi = FeatureExtractor(pretrained=True).to(DEVICE)
_G   = Classifier(_phi.out_dim, NUM_CLASSES).to(DEVICE)
_pt, _Gt = create_teacher(_phi, _G, NUM_CLASSES, DEVICE)

_before_w  = _Gt.fc.weight.clone()
_before_bn = _pt.net[1].running_mean.clone()

with torch.no_grad():
    _G.fc.weight.add_(1.0)                       # giả lập student đã học
update_ema([_phi, _G], [_pt, _Gt], global_step=1)

_pt.train()
with torch.no_grad():
    _pt(torch.randn(4, 3, 224, 224, device=DEVICE))

print('teacher weight da doi   :', not torch.allclose(_before_w,  _Gt.fc.weight))
print('teacher BN buffer da doi:', not torch.allclose(_before_bn, _pt.net[1].running_mean))
del _phi, _G, _pt, _Gt
```

Expected:
```
teacher weight da doi   : True
teacher BN buffer da doi: True
```

Cả hai phải là `True`. Nếu dòng thứ hai là `False`, teacher đang ở `eval()` — dừng lại và sửa.

- [ ] **Step 4: Commit**

```bash
git add Plant_Disease_UDA_Final.ipynb
git commit -m "feat: add unified train_cdane(seed, use_mt) for both experiment branches"
```

---

### Task 6: Runner đa seed

**Files:**
- Modify: `Plant_Disease_UDA_Final.ipynb` (chèn 1 cell sau cell `train_cdane`)

**Interfaces:**
- Consumes: `train_cdane` (Task 5)
- Produces:
  - `MT_RESULTS_CSV: str` — đường dẫn file CSV kết quả
  - `run_experiment_grid(seeds=(1,2,3,4,5), branches=(False, True)) -> pandas.DataFrame` với các cột `seed`, `use_mt`, `acc_selected`, `acc_student_final`, `best_val_loss`, `epochs_run`

- [ ] **Step 1: Chèn cell code runner**

Chèn ngay sau cell `train_cdane`. Loại cell: `code`.

```python
# === CELL-TAG: run_grid ===
# ══════════════════════════════════════════════════════════════════════════════
# RUNNER ĐA SEED
#
# 10 run x 300 epoch. Colab sẽ ngắt session giữa chừng, nên mỗi run vừa xong là
# ghi ngay vào CSV trên Drive; chạy lại cell này sẽ bỏ qua những run đã có.
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd

MT_RESULTS_CSV = os.path.join(SAVE_DIR, 'cdane_mt_results.csv')


def _load_results():
    if os.path.exists(MT_RESULTS_CSV):
        return pd.read_csv(MT_RESULTS_CSV)
    return pd.DataFrame(columns=['seed', 'use_mt', 'acc_selected',
                                 'acc_student_final', 'best_val_loss',
                                 'epochs_run'])


def run_experiment_grid(seeds=(1, 2, 3, 4, 5), branches=(False, True)):
    """Chạy toàn bộ lưới thí nghiệm, bỏ qua những ô đã hoàn thành.

    Baseline chạy trước cho cả 5 seed rồi mới tới MT, để nếu Colab ngắt sớm thì
    ít nhất cũng có một nhánh trọn vẹn.

    Returns:
        DataFrame toàn bộ kết quả đã có.
    """
    df = _load_results()
    for use_mt in branches:
        for seed in seeds:
            done = ((df['seed'] == seed) & (df['use_mt'] == use_mt)).any()
            if done:
                print(f'[skip] seed={seed} use_mt={use_mt} — da co ket qua')
                continue

            label = 'CDAN+E + MT' if use_mt else 'CDAN+E'
            print(f'\n{"="*60}\n{label}  |  seed={seed}\n{"="*60}')
            res = train_cdane(seed=seed, use_mt=use_mt, ckpt_dir=SAVE_DIR)

            df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
            df.to_csv(MT_RESULTS_CSV, index=False)   # ghi ngay, đừng đợi hết vòng
            print(f'  -> acc_selected = {res["acc_selected"]:.2f}%  [da luu CSV]')
    return df


print(f'Runner ready. Ket qua se ghi vao:\n  {MT_RESULTS_CSV}')
```

- [ ] **Step 2: Chạy thử lưới rút gọn trên Colab**

Trước khi cam kết 10 run dài, xác minh cơ chế bỏ qua và ghi CSV hoạt động, bằng một lưới 2 ô ngắn.

```python
_probe = os.path.join(SAVE_DIR, 'cdane_mt_results.csv')
assert not os.path.exists(_probe), 'Xoa CSV cu truoc khi chay thu'

df_probe = run_experiment_grid(seeds=(1,), branches=(False, True))
print(df_probe)
df_again = run_experiment_grid(seeds=(1,), branches=(False, True))   # phải skip cả 2
```

Expected: lần gọi thứ hai in ra hai dòng `[skip]` và không huấn luyện gì thêm.

Sau khi xác minh xong, xoá CSV thử để chạy thật:
```python
os.remove(MT_RESULTS_CSV)
```

*Lưu ý:* run thật dùng mặc định `max_epochs=300`, nên bước thử này cũng tốn 2 run đầy đủ. Nếu muốn thử nhanh, tạm truyền `max_epochs=4, min_sel=3` vào `train_cdane` bên trong `run_experiment_grid` bằng cách gọi trực tiếp `train_cdane` thay vì qua runner, rồi kiểm tra logic skip riêng.

- [ ] **Step 3: Chạy lưới đầy đủ trên Colab**

```python
df = run_experiment_grid(seeds=(1, 2, 3, 4, 5), branches=(False, True))
df
```

Chạy nhiều session nếu cần — cell này resume được ở hai cấp: bỏ qua run đã xong (CSV) và tiếp tục run dở (checkpoint).

Expected: 10 dòng trong `df`, mỗi dòng có `acc_selected` trong khoảng hợp lý (60–95%).

- [ ] **Step 4: Commit**

```bash
git add Plant_Disease_UDA_Final.ipynb
git commit -m "feat: add resumable multi-seed experiment runner"
```

---

### Task 7: Thống kê và bảng kết quả

**Files:**
- Modify: `Plant_Disease_UDA_Final.ipynb` (chèn 1 cell sau cell `run_grid`)

**Interfaces:**
- Consumes: `MT_RESULTS_CSV`, `_load_results` (Task 6); `all_results`, `paper_results` (cell sẵn có)
- Produces: `summarize_mt_experiment() -> pandas.DataFrame` — in bảng tóm tắt, chạy paired t-test, và ghi hai khoá `'CDAN+E (5 seed)'`, `'CDAN+E + MT (5 seed)'` vào `all_results`

- [ ] **Step 1: Chèn cell code thống kê**

Chèn ngay sau cell `run_grid`. Loại cell: `code`.

```python
# === CELL-TAG: mt_stats ===
# ══════════════════════════════════════════════════════════════════════════════
# THỐNG KÊ
#
# Câu hỏi cần trả lời KHÔNG phải "có vượt 91.1% của paper không".
# Reproduce hiện tại vượt paper ở 5/6 method (DANN +19.9, DALN +23.2), nên split
# đang dùng dễ hơn setting của paper một cách hệ thống — vượt 91.1% có thể xảy
# ra mà chẳng chứng minh MT đóng góp gì.
#
# Câu hỏi đúng: CDAN+E+MT có hơn CDAN+E của CHÍNH MÌNH không, cùng split, cùng
# seed, cùng protocol. Vì hai nhánh ghép cặp theo seed, paired t-test mạnh hơn
# so sánh hai mean rời rạc.
# ══════════════════════════════════════════════════════════════════════════════

from scipy import stats


def summarize_mt_experiment():
    """In bảng tóm tắt, chạy paired t-test, cập nhật all_results."""
    df = _load_results()
    if df.empty:
        print('Chua co ket qua. Chay run_experiment_grid() truoc.')
        return df

    base = df[~df['use_mt'].astype(bool)].sort_values('seed')
    mt   = df[ df['use_mt'].astype(bool)].sort_values('seed')

    print('=' * 68)
    print(f'{"seed":>6}{"CDAN+E":>14}{"CDAN+E + MT":>16}{"delta":>12}')
    print('-' * 68)

    shared = sorted(set(base['seed']) & set(mt['seed']))
    b_vals, m_vals = [], []
    for s in shared:
        b = float(base.loc[base['seed'] == s, 'acc_selected'].iloc[0])
        m = float(mt.loc[mt['seed'] == s, 'acc_selected'].iloc[0])
        b_vals.append(b); m_vals.append(m)
        print(f'{s:>6}{b:>13.2f}%{m:>15.2f}%{m - b:>+11.2f}%')

    print('-' * 68)
    if b_vals:
        bm, bs = np.mean(b_vals), np.std(b_vals, ddof=1) if len(b_vals) > 1 else 0.0
        mm, ms = np.mean(m_vals), np.std(m_vals, ddof=1) if len(m_vals) > 1 else 0.0
        print(f'{"mean":>6}{bm:>12.2f}%{mm:>14.2f}%{mm - bm:>+11.2f}%')
        print(f'{"std":>6}{bs:>12.2f} {ms:>14.2f} ')

        # Guard: all_results được tạo ở cell Baseline. Nếu chưa chạy cell đó
        # thì tạo mới, để không mất kết quả sau một run dài chỉ vì NameError.
        if 'all_results' not in globals():
            all_results = {}
        all_results['CDAN+E (5 seed)']      = bm
        all_results['CDAN+E + MT (5 seed)'] = mm

    print('=' * 68)

    if len(shared) >= 3:
        t, p = stats.ttest_rel(m_vals, b_vals)
        print(f'Paired t-test (n={len(shared)}): t={t:.3f}  p={p:.4f}')
        if p < 0.05:
            verdict = 'CO cai thien' if np.mean(m_vals) > np.mean(b_vals) else 'LAM TE DI'
            print(f'  -> p < 0.05: Mean Teacher {verdict} co y nghia thong ke.')
        else:
            print('  -> p >= 0.05: khong du bang chung ket luan MT co tac dung.')
            print('     Ket qua am tinh nay VAN dang bao cao trong luan van.')
    else:
        print(f'Moi co {len(shared)} cap seed — can it nhat 3 de chay t-test.')

    print()
    print('Tham chieu ngoai (KHONG phai moc so sanh):')
    print(f'  Paper CDAN+E w/ FBR (Apple) : 91.1 +- 4.22%')
    print(f'  Paper Real-to-Real (tran tren): 97.7 +- 0.77%')
    print(f'  Cell 42, 1 seed, target khong augment: 87.28%')
    print()
    print('LUU Y: baseline o day khong bang 87.28% cua cell 42, vi anh target')
    print('gio co augmentation (bat buoc de Mean Teacher hoat dong). Ca hai')
    print('nhanh dung chung pipeline target nen so sanh giua chung van hop le.')

    return df


summary_df = summarize_mt_experiment()
summary_df
```

- [ ] **Step 2: Chạy trên Colab**

Chạy cell vừa chèn.

Expected: bảng 5 dòng seed, dòng mean/std, kết quả t-test với giá trị p, và khối tham chiếu.

- [ ] **Step 3: Chạy lại cell bảng kết quả tổng**

Chạy lại cell chứa `order = ['Baseline', 'FBR', ...]` (mục `## 9 — Results Comparison`). Hai khoá mới đã có trong `all_results` nên sẽ hiện ra ở cuối bảng với cột `Paper` là `—`.

Nếu muốn chúng nằm đúng thứ tự, sửa danh sách `order` trong cell đó thành:

```python
order = ['Baseline', 'FBR', 'DDC w/ FBR', 'DCORAL w/ FBR',
         'DANN w/ FBR', 'CDAN+E w/ FBR', 'DALN w/ FBR',
         'CDAN+E (5 seed)', 'CDAN+E + MT (5 seed)']
```

Đây là lần duy nhất trong kế hoạch có sửa một cell đã tồn tại, và chỉ sửa đúng một danh sách.

- [ ] **Step 4: Commit**

```bash
git add Plant_Disease_UDA_Final.ipynb
git commit -m "feat: add paired t-test summary for CDAN+E vs CDAN+E+MT"
```

---

## Việc cần làm ngoài kế hoạch này

Ghi lại ở đây để không bị quên; không nằm trong phạm vi các task trên.

1. **Thu hồi Kaggle API token.** Cell 9 hardcode `KAGGLE_API_TOKEN = 'KGAT_f145b2ef3e319f8898282f98be50beef'`. Notebook này nếu nộp kèm luận văn hoặc chia sẻ là lộ credential. Thu hồi token trên kaggle.com rồi chuyển sang upload `kaggle.json`, đúng như markdown cell 8 đã khuyến cáo. **Token đã bị commit vào git ở Task 1, nên việc thu hồi là bắt buộc, không phải tuỳ chọn.**

2. **Ba cell DALN trùng nhau.** Cell 44, 45, 46 cùng ghi vào `all_results['DALN w/ FBR']` với ba kết quả khác nhau (82.51 / 43.21 / 51.30). Bảng kết quả lấy giá trị của cell chạy sau cùng. Nên xoá hai cell thừa, giữ lại cấu hình đúng.
