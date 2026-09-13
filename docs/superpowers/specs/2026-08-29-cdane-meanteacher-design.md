# Thiết kế: Kết hợp CDAN+E và Mean Teacher cho UDA chẩn đoán bệnh cây

**Ngày:** 2026-08-29
**Bài toán:** PlantVillage (lab, có nhãn) → PlantPathology (field, không nhãn), Apple, 3 lớp
**Mục tiêu:** Xác định xem consistency regularization kiểu Mean Teacher có cải thiện CDAN+E trong bối cảnh UDA hay không, với bằng chứng thống kê đủ mạnh.

---

## 1. Bối cảnh

### 1.1. Hai nguồn tham chiếu

**Paper A** — Jeon et al., *"Bridging the Lab-to-Field gap in plant disease diagnosis through unsupervised domain adaptation enhanced by background recomposition"*, Ecological Informatics 2026.
Reproduce trong `Plant_Disease_UDA_Final.ipynb`. Pipeline: FBR (SAM segment lá + ghép nền field) → UDA (DDC, DCORAL, DANN, CDAN+E, DALN). Backbone ResNet-18, AdamW lr=1e-3, cosine annealing, ảnh 224x224, batch 64.

**Paper B** — Ilsever & Baz, *"Consistency regularization based semi-supervised plant disease recognition"*, Smart Agricultural Technology 9 (2024) 100613.
Reproduce trong `Reproduce_Ilsever2024_MeanTeacher_Colab.ipynb`, chạy trực tiếp code gốc tại `github.com/milsever/plant-pathology`. Setting: semi-supervised một domain trên PP2021, ResNet-50, HeadThenBody fine-tuning.

### 1.2. Kết quả reproduce hiện có (Apple, genuine setting)

| Method | Repro (1 seed) | Paper (5 seed, mean ± std) | Delta |
|---|---|---|---|
| Baseline | 60.40 | 47.1 ± 6.54 | +13.30 |
| w/ FBR | 67.05 | 57.0 ± 7.01 | +10.05 |
| DDC w/ FBR | 75.14 | 57.7 ± 3.63 | +17.44 |
| DCORAL w/ FBR | 79.77 | 61.0 ± 3.68 | +18.77 |
| DANN w/ FBR | 92.34 | 72.4 ± 7.50 | +19.94 |
| **CDAN+E w/ FBR** | **87.28** | **91.1 ± 4.22** | **−3.82** |
| DALN w/ FBR | 82.51 | 59.3 ± 1.77 | +23.21 |
| Real-to-Real (trần trên) | — | 97.7 ± 0.77 | — |

Split hiện dùng: PVD train 656 / PVD val 219 / PPD target 1038 (không nhãn) / PPD test 692.

### 1.3. Hai quan sát định hướng toàn bộ thiết kế

**(a) CDAN+E là method duy nhất reproduce thấp hơn paper.** Đây chính là khoảng trống mà consistency regularization có thể lấp.

**(b) Setup reproduce mạnh hơn paper một cách hệ thống.** Năm trong sáu method vượt paper từ +10 đến +23 điểm; DANN đạt 92.34% trong khi paper chỉ 72.4%. Điều này chứng tỏ split PPD / subset 3 lớp đang dùng dễ hơn setting của paper.

**Hệ quả bắt buộc:** con số 91.1% của paper **không phải** là mốc so sánh có giá trị khoa học. Vượt nó có thể xảy ra mà không chứng minh được MT đóng góp gì. Phép so sánh hợp lệ duy nhất là:

> **CDAN+E + MT vs CDAN+E baseline của chính mình, cùng split, cùng seed, cùng protocol model selection.**

Con số paper chỉ được ghi làm tham chiếu ngoài.

### 1.4. Cơ sở lý thuyết của việc kết hợp

Mean Teacher áp dụng cho UDA đã có tiền lệ chuẩn: **French, Mackiewicz & Fisher, "Self-Ensembling for Visual Domain Adaptation", ICLR 2018** — teacher EMA + consistency loss trên target domain không nhãn.

Hai cơ chế bổ sung nhau, không chồng lấn:
- **CDAN+E** làm *feature alignment có điều kiện theo lớp* giữa hai domain (multilinear map giữa feature và prediction).
- **Mean Teacher** làm *ổn định quyết định trên target* (smoothness assumption quanh mỗi mẫu target).

---

## 2. Các quyết định thiết kế và lý do

| # | Quyết định | Lựa chọn | Lý do |
|---|---|---|---|
| D1 | Phạm vi thí nghiệm | 5 seed, chỉ Apple | Đúng protocol paper (5 lần lặp). 87.28 vs 91.1 ± 4.22 nằm trong 1 std, nên 1 seed không kết luận được gì. |
| D2 | Kiến trúc kết hợp | Joint 1 giai đoạn | Đúng với French et al. ICLR 2018 và là cách chuẩn trong literature. |
| D3 | Augmentation 2 view | Weak/weak | Ilsever & Baz thử cả weak và strong trên **chính dataset lá cây này** và weak thắng. Strong color distortion có thể xoá dấu hiệu phân biệt lớp (rust=cam/nâu, scab=nâu/đen, healthy=xanh). |
| D4 | Lịch ramp consistency | Ramp trễ, epoch 100→150 | UDA có ràng buộc mà SSL không có: λ_GRL cũng đang ramp. Ép consistency sớm khi feature còn xáo trộn mạnh dễ gây confirmation bias. |
| D5 | Model selection | Y nguyên protocol baseline, áp lên teacher | Giữ tính công bằng tuyệt đối: hai nhánh chỉ khác nhau đúng một biến. |

---

## 3. Kiến trúc

```
student:  phi (ResNet-18, ImageNet init)  →  G (Linear 512 → 3)    [có gradient]
teacher:  phi'                            →  G'                    [KHÔNG gradient, = EMA(student)]
D:        CDANDisc(in_dim=512, num_classes=3)                       [giữ nguyên cell 32]
GRL:      GradRevLayer                                              [giữ nguyên cell 32]
```

Teacher chỉ có hai vai trò: sinh mục tiêu cho consistency loss, và là model đem đi đánh giá cuối cùng. Nó không bao giờ nhận gradient.

Tái sử dụng nguyên vẹn từ cell 32: `FeatureExtractor`, `Classifier`, `CDANDisc`, `multilinear_map`, `GRL`, `GradRevLayer`, `get_lambda`, `domain_lbl`, `uda_test`.

---

## 4. Luồng dữ liệu

| Loader | Nguồn | Số ảnh | Transform | Nhãn |
|---|---|---|---|---|
| `train_loader_A_fbr` | FBR source | 656 | `TRAIN_TRANSFORM` | Có |
| `target_loader_2view` | PPD target | 1038 | `TwoCropTransform(TRAIN_TRANSFORM)` | Không |
| `val_loader_A_fbr` | FBR val | 219 | `VAL_TRANSFORM` | Có (source) |
| `test_loader_B` | PPD test | 692 | `VAL_TRANSFORM` | Có (chỉ để đo) |

Chỉ thêm duy nhất `TwoCropTransform`. Không đụng vào dataset, split, hay `TRAIN_TRANSFORM` — mọi thay đổi ở đây sẽ phá tính so sánh với 87.28%.

```python
class TwoCropTransform:
    """Trả về 2 view độc lập của cùng một ảnh. Nguồn ngẫu nhiên duy nhất
    là 2 lần bốc mẫu khác nhau của RandomCrop / RandomHorizontalFlip / ColorJitter."""
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, img):
        return self.transform(img), self.transform(img)
```

Batch size giữ 64 cho cả hai loader, `zip(train_loader_A_fbr, target_loader_2view)` như cell 42.

---

## 5. Hàm mục tiêu

```
L = L_cls + L_cdan + w(t) · L_cons
```

### 5.1. `L_cls` và `L_cdan` — sao chép nguyên xi cell 42

Không sửa logic nào. Adversarial loss chạy trên `src` và `tgt_v2`.

**Quy ước hai view** (cả hai đều weak, theo D3 — hậu tố chỉ là số thứ tự, không phải cường độ):

| View | Ai forward | Dùng cho |
|---|---|---|
| `tgt_v1` | teacher | mục tiêu của `L_cons` |
| `tgt_v2` | student | **cả** `L_cdan` **và** `L_cons` |

Student chỉ forward target **một lần** mỗi step: feature `tf` dùng chung cho adversarial loss và consistency loss. Nhờ vậy chi phí mỗi step chỉ tăng đúng một forward không-gradient của teacher so với cell 42, chứ không phải hai forward.

```python
sf = phi(src);  tf = phi(tgt_v2)
logits_s = G(sf)
sp = softmax(logits_s);  tp = softmax(G(tf))

L_cls = ce(logits_s, sl)

src_mm = grl(multilinear_map(sf, sp.detach()))   # detach PRED, giữ gradient FEATURE
tgt_mm = grl(multilinear_map(tf, tp.detach()))

sw = cdan_entropy_weight(sp); sw = sw / sw.sum()  # w = 1 + exp(-H(p)), chuẩn hoá sum=1
tw = cdan_entropy_weight(tp); tw = tw / tw.sum()

src_d = (sw * ce_persam(D(src_mm), domain_lbl(n_s, True,  dev))).sum()
tgt_d = (tw * ce_persam(D(tgt_mm), domain_lbl(n_t, False, dev))).sum()
L_cdan = (src_d + tgt_d) / 2.0
```

**Cảnh báo đã ghi trong markdown cell 41:** không được detach `sf` / `tf` trong multilinear map, nếu không gradient không tới được `phi` qua GRL và accuracy sụp về ~48%.

### 5.2. `L_cons` — dùng đúng công thức repo gốc

Từ `losses.py` của `milsever/plant-pathology` (bản thân nó dẫn nguồn `CuriousAI/mean-teacher`):

```python
def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax  = F.softmax(input_logits,  dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes    = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes
```

Cách gọi (theo `ft_meanteacher.py`: `consistency_weight * criterion(out, ema_out) / batch_size`):

```python
with torch.no_grad():
    teacher_logits = G_t(phi_t(tgt_v1))         # teacher: view 1
student_logits = G(tf)                          # student: tái dùng tf = phi(tgt_v2) ở 5.1
L_cons = softmax_mse_loss(student_logits, teacher_logits) / tgt_v2.size(0)
```

Lưu ý `student_logits` ở đây chính là `G(tf)`, tức đúng logits đã tính ở 5.1 để lấy `tp`. Trong code thực tế chỉ cần giữ lại biến đó, không gọi lại `G`.

**Về việc detach:** `losses.py` gốc **không** detach `target_logits`. Repo chặn gradient bằng `self.ema_model.freeze_all()` khi khởi tạo. Cell 6 của notebook B chính là để kiểm chứng điều này bằng thực nghiệm. Ta làm cả hai cho chắc: `requires_grad_(False)` trên toàn bộ tham số teacher **và** bọc forward trong `torch.no_grad()`.

---

## 6. Cập nhật EMA

Sao chép chính xác `update_ema_variables` của official:

```python
def update_ema(student, teacher, global_step, ema_decay=0.99):
    # Dùng true average cho đến khi exponential average đủ chính xác
    alpha = min(1 - 1 / (global_step + 1), ema_decay)
    for ema_p, p in zip(teacher.parameters(), student.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)
```

`ema_decay = 0.99` — giá trị paper B dùng, lấy từ thí nghiệm "4000 labels / ResNet on CIFAR-10" của paper MT gốc.

### 6.1. Cạm bẫy BatchNorm — điểm dễ sai nhất của cả thiết kế

`.parameters()` **không** bao gồm `running_mean` / `running_var` của BatchNorm; chúng là *buffers*. ResNet-18 có 20 lớp BN.

Official không EMA các buffer này. Cách nó hoạt động: **teacher được để ở chế độ `.train()` trong suốt quá trình huấn luyện**, nên BN running stats của teacher tự cập nhật từ chính forward pass của nó trên dữ liệu target.

Quy tắc bắt buộc:

| Giai đoạn | Chế độ teacher | Bọc no_grad |
|---|---|---|
| Training (forward `tgt_v1`) | `.train()` | Có |
| Validation | `.eval()` | Có |
| Test | `.eval()` | Có |

Nếu để teacher ở `.eval()` lúc training, nó sẽ dùng BN stats khởi tạo từ ImageNet suốt 300 epoch và kết quả sẽ tệ thảm hại.

### 6.2. Khởi tạo teacher

```python
phi_t = FeatureExtractor(pretrained=True).to(DEVICE)
G_t   = Classifier(phi_t.out_dim, NUM_CLASSES).to(DEVICE)
phi_t.load_state_dict(phi.state_dict())     # bắt đầu giống hệt student
G_t.load_state_dict(G.state_dict())
for p in list(phi_t.parameters()) + list(G_t.parameters()):
    p.requires_grad_(False)
```

---

## 6.3. Optimizer

Giữ nguyên hoàn toàn cấu hình của cell 42 để hai nhánh chỉ khác nhau đúng một biến:

```python
opt = optim.AdamW(list(phi.parameters()) + list(G.parameters()) + list(D.parameters()),
                  lr=1e-3, weight_decay=0.01)
sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
```

Tham số của teacher **không** nằm trong optimizer — nó chỉ được cập nhật bằng EMA ở mục 6.

---

## 7. Hai lịch ramp

### 7.1. λ_GRL — giữ nguyên

`get_lambda(step, total_steps, gamma=10.0)` từ cell 32. Không đổi.

### 7.2. Consistency weight

```python
def sigmoid_rampup(x, length):                      # ramps.py gốc
    if length == 0:
        return 1.0
    x = float(np.clip(x, 0.0, length)) / length
    return float(np.exp(-5.0 * (1.0 - x) ** 2))

CONS_MAX        = 9.0    # = C^2, theo quy tắc [C, C^2] của paper B (C=3)
CONS_START      = 100    # epoch bắt đầu ramp
CONS_RAMP_LEN   = 50     # epoch để bão hoà

def consistency_weight(epoch):
    if epoch < CONS_START:
        return 0.0
    return CONS_MAX * sigmoid_rampup(epoch - CONS_START, CONS_RAMP_LEN)
```

**Nguồn của `CONS_MAX = 9`:** paper B chọn consistency weight trong khoảng `[C, C²]`; với 6 lớp khoảng đó là [6,36] và họ lấy 30 (gần cận trên). Với 3 lớp, khoảng là [3,9] → lấy 9.

**Nguồn của lịch ramp trễ:** paper B ramp trong 5/90 epoch (5.6%), quy sang 300 epoch là ~17 epoch. Ta lệch khỏi con số đó **có chủ ý** vì UDA có λ_GRL đồng thời đang ramp. Đây là một điều chỉnh có lý do rõ ràng và nên được viết vào luận văn như một đóng góp.

| epoch | λ_GRL | w_cons |
|---|---|---|
| 1 | 0.00 | 0 |
| 50 | 0.76 | 0 |
| 100 | 0.93 | 0 |
| 125 | 0.99 | 4.5 |
| 150 | 1.00 | 9 |
| 250–300 | 1.00 | 9 |

---

## 8. Model selection và đánh giá

Giống hệt cell 42, chỉ đổi đối tượng từ student sang teacher:

```python
MAX_EPOCHS = 300
MIN_SEL    = 250

phi_t.eval(); G_t.eval()
# ... tính val_loss của TEACHER trên val_loader_A_fbr ...
if epoch >= MIN_SEL and val_loss_teacher < best_loss:
    best_loss  = val_loss_teacher
    best_state = {'phi': copy.deepcopy(phi_t.state_dict()),
                  'G':   copy.deepcopy(G_t.state_dict())}
phi_t.train(); G_t.train()      # trả lại train mode cho BN (mục 6.1)
```

Protocol này hợp lệ vì `w_cons` đã bão hoà từ epoch 150. Cảnh báo của paper B ("tắt early stopping cho MT vì val loss tăng trong lúc ramp-up") chỉ áp dụng cho cửa sổ ramp, không áp dụng cho cửa sổ chọn model 250–300.

**Test:** teacher trên `test_loader_B`. **Đồng thời log accuracy của student** — nếu student tốt hơn teacher rõ rệt thì gần như chắc chắn đã sai ở mục 6.1.

---

## 9. Ma trận thí nghiệm

| Nhánh | Seed | Epoch | Ghi chú |
|---|---|---|---|
| CDAN+E (baseline) | 1, 2, 3, 4, 5 | 300 | Phải chạy lại; hiện chỉ có 1 seed. Gọi `set_seed(s)` đầu mỗi run. |
| CDAN+E + MT | 1, 2, 3, 4, 5 | 300 | Cùng seed để ghép cặp |

Tổng 10 run × 300 epoch ResNet-18. Checkpoint mỗi 10 epoch vào Drive (tái dùng `save_uda_checkpoint` / `load_uda_checkpoint` của cell 34, mở rộng để lưu thêm state của teacher).

**Báo cáo:**
- mean ± std cho mỗi nhánh
- **paired t-test theo seed** (mạnh hơn so sánh hai mean rời rạc, vì cùng seed loại được biến thiên do khởi tạo)
- tham chiếu ngoài: paper 91.1 ± 4.22; trần trên Real-to-Real 97.7

---

## 10. Phát hiện hỏng — log mỗi 10 epoch

| Triệu chứng | Nguyên nhân khả dĩ |
|---|---|
| Teacher tệ hơn student rõ rệt | BN buffer chưa sync → sai mục 6.1 |
| Phân bố argmax của teacher trên target lệch >70% về một lớp | Class collapse / confirmation bias → hạ `CONS_MAX` hoặc lùi `CONS_START` |
| `L_cons` ≈ 0 ngay từ epoch 100 | Hai view quá giống nhau, hoặc EMA decay quá cao |
| Val loss vọt lên quanh epoch 100 | **Bình thường** — đúng lúc bắt đầu ramp |
| Accuracy sụp về ~48% | Gradient không tới `phi` qua GRL (lỗi đã ghi ở markdown cell 41) |
| `L_cons` tăng đơn điệu không giảm | `CONS_MAX` quá cao so với `L_cls` + `L_cdan` |

Đại lượng cần log: `L_cls`, `L_cdan`, `L_cons`, `w(t)`, `λ_GRL`, val_acc teacher, val_acc student, phân bố argmax teacher trên target.

---

## 11. Bố cục code

Chèn 5 cell mới **sau** cell 42. Không sửa cell nào đang có (cell 42 giữ nguyên làm baseline).

| Cell | Nội dung |
|---|---|
| 8.6b | `TwoCropTransform`, `target_loader_2view` |
| 8.6c | `softmax_mse_loss`, `sigmoid_rampup`, `consistency_weight`, `update_ema`, `create_teacher` |
| 8.6d | Vòng huấn luyện CDAN+E + MT cho **một** seed (có checkpoint/resume) |
| 8.6e | Runner đa seed: chạy 5 seed cho cả hai nhánh, gom kết quả |
| 8.6f | Thống kê: mean ± std, paired t-test, bảng so sánh |

Cell 48 (bảng kết quả) mở rộng thêm dòng `CDAN+E + MT w/ FBR`.

---

## 12. Ngoài phạm vi nhưng cần xử lý

1. **Credential bị lộ.** Cell 9 của `Plant_Disease_UDA_Final.ipynb` hardcode Kaggle API token (`KAGGLE_API_TOKEN = 'KGAT_...'`). Nên thu hồi token đó và chuyển sang upload `kaggle.json` — đúng như markdown cell 8 đã khuyến cáo. Không sửa trong phạm vi công việc này nhưng phải làm trước khi chia sẻ notebook.

2. **Ba cell DALN trùng nhau.** Cell 44, 45, 46 cùng ghi vào `all_results['DALN w/ FBR']` với ba kết quả khác nhau (82.51 / 43.21 / 51.30). Bảng cell 48 sẽ lấy giá trị của cell chạy sau cùng. Không ảnh hưởng đến việc ghép MT, nhưng sẽ gây nhầm khi báo cáo.

3. **Git chưa được khởi tạo** trong thư mục này, nên spec chưa được commit.
