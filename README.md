# Occupant-Centric HVAC Optimization — Bayesian Preference Learning + PPO (GRU surrogate)

Khung điều khiển HVAC cá nhân hóa cho khí hậu nóng ẩm Việt Nam: học sở thích nhiệt độ/độ ẩm cá nhân (T\*, RH\*) bằng **Bayesian inference**, dùng làm reward cá nhân hóa để huấn luyện **PPO** trên một **surrogate environment** (Dual-GRU) học từ dữ liệu cảm biến thật (CAMaRSEC, 49 căn hộ Hà Nội).

Repo này chứa **code + dữ liệu + model đã train + kết quả đánh giá thật** — không phải khung sườn, toàn bộ pipeline dưới đây đã chạy được end-to-end (đã verify lại trước khi đóng gói).

---

## Trạng thái từng Phase

| Phase | Thành phần | File | Trạng thái |
|---|---|---|---|
| 1 | Data prep | `preprocess_chinese_comfort.py`, `data/` | ✅ |
| 2 | Environment Model (GRU) | `Env_model/dual_gru_env_model.py` + weights | ✅ Đã train (holdout R² T≈0.93/0.87, RH≈0.76/0.67) |
| 3 | Bayesian Preference | `bayesian_learning.py`, `notebooks/phase3_summary.ipynb` | ✅ |
| 4 | Reward Function | `hvac_reward_env.py` | ✅ (canonical — xem "Reward: các phiên bản" bên dưới) |
| 5 | Gym Environment | `env_wrapper.py` | ✅ **Mới viết + đã test** (không có trong bản gốc, được tái tạo khớp interface mà Phase 6/7 kỳ vọng) |
| 6 | RL Training (PPO) | `ppo_hvac.zip` | ✅ Checkpoint đã train (script train gốc chưa có, không bắt buộc để dùng lại model) |
| 7 | Evaluation & Baselines | `phase7_evaluation_update.py` + `results/` | ✅ Đã chạy, kèm exploitation self-check |
| 8 | Real-world Pilot | — | ⏳ Chưa triển khai (hướng mở rộng) |

**Env model dùng cho toàn bộ pipeline là GRU** (`Env_model/dual_gru_env_model.py`). Có một bản LightGBM thử nghiệm (`Env_model/experimental/dual_lgbm_env_model_v2.py`) nhưng **không được dùng** ở Phase 5/6/7 — giữ lại chỉ để tham khảo.

---

## ⚠️ Lưu ý quan trọng trước khi dùng kết quả

1. **`env_wrapper.py` là file do Claude viết lại**, không phải bản gốc của bạn (bản gốc không được upload). Nó đã được test bằng `gymnasium.utils.env_checker.check_env`, chạy với model GRU thật, và chạy trọn `phase7_evaluation_update.py` (baseline + PPO checkpoint) ra kết quả khớp format các file bạn từng có. Tuy nhiên **logic bên trong (cách advance state bằng forecast t+30, cách sample trajectory theo từng apartment...) là suy luận hợp lý từ interface, không phải bản gốc — nên review kỹ trước khi dùng cho kết quả công bố chính thức.**
2. **`exploitation_check.json`** (kết quả gốc bạn upload) từng báo `exploitation_suspected: true` (RL energy chỉ 38% so với Rule_Bayesian). Repo này **chưa xử lý vấn đề đó** — theo yêu cầu, phần điều tra nguyên nhân bị bỏ qua ở bước này. Đừng công bố kết quả RL_PPO cho đến khi vấn đề này được làm rõ.
3. **Thiếu 2 file output của Phase 3** để cá nhân hóa đầy đủ khi chạy lại `phase7_evaluation_update.py`:
   - `outputs/bayesian_preference/comfort_gaussian_params.json`
   - `bayesian_preference_results.csv` (T\*/RH\* theo từng apartment)

   Hai file này phải được **tạo lại bằng `bayesian_learning.py`** trước khi chạy evaluation. Nếu không có, `hvac_reward_env.py` sẽ tự dùng posterior mặc định ("Run C" hard-coded).
4. **`results/evaluation_config.json`** (file gốc) ghi `"env_model_path": "Env_model/env_model_gru"` — đường dẫn này **không khớp** với cấu trúc thực tế (model nằm trực tiếp trong `Env_model/`, không có thư mục con `env_model_gru`). Dùng `--env-model-path Env_model` khi chạy (đã verify đúng).

---

## Reward: các phiên bản

| File | Vai trò |
|---|---|
| `hvac_reward_env.py` | **Canonical** — được `env_wrapper.py` và `phase7_evaluation_update.py` import trực tiếp (`ComfortModel`, `load_comfort_model`, `reward_from_env_model`) |
| `archive/hvac_reward_v1.py` | Bản đầu tiên — chỉ giữ tham khảo |
| `archive/hvac_reward_v2.py` | Bản trung gian trước `hvac_reward_env.py` — chỉ giữ tham khảo |

---

## Cấu trúc thư mục

```
hvac-preference-rl/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── docs/
│   ├── research_proposal.md
│   └── pipeline.html
│
├── data/
│   ├── raw/                 # CAMaRSEC zip, Chinese Comfort Class I/II/III (gitignored — quá lớn)
│   └── processed/            # final_dataset_with_setpoint_proxy.csv, filtered comfort csv (gitignored)
│
├── Env_model/                # Phase 2 — canonical GRU surrogate
│   ├── dual_gru_env_model.py
│   ├── gru_T.pt, gru_RH.pt, scaler_T.pkl, scaler_RH.pkl, meta.json, training_report.json
│   └── experimental/
│       └── dual_lgbm_env_model_v2.py   # KHÔNG dùng trong pipeline chính
│
├── preprocess_chinese_comfort.py   # Phase 1
├── bayesian_learning.py            # Phase 3
├── hvac_reward_env.py              # Phase 4 (canonical)
├── env_wrapper.py                  # Phase 5 (mới, đã test)
├── ppo_hvac.zip                    # Phase 6 (checkpoint đã train)
├── phase7_evaluation_update.py     # Phase 7 (canonical)
│
├── archive/                        # các phiên bản reward cũ
├── notebooks/
│   └── phase3_summary.ipynb
├── results/                         # kết quả evaluation đã có sẵn (evaluation_summary.csv, exploitation_check.json...)
└── configs/
```

---

## Cài đặt

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Chạy lại Phase 7 evaluation (đã verify chạy được với lệnh này)

```bash
# 1. (Bắt buộc) Tạo lại output Phase 3 nếu chưa có:
python bayesian_learning.py   # xem docstring trong file để biết đúng tham số/đường dẫn output

# 2. Chạy evaluation — baseline + PPO checkpoint có sẵn
python phase7_evaluation_update.py \
  --env-model-path Env_model \
  --comfort-model outputs/bayesian_preference/comfort_gaussian_params.json \
  --preference-csv bayesian_preference_results.csv \
  --results results/run1 \
  --episodes 3 \
  --policy ppo_hvac.zip
```

Nếu chỉ muốn chạy baseline (không cần PPO/Bayesian output), bỏ `--policy` và `--comfort-model`/`--preference-csv` — `hvac_reward_env.py` sẽ tự dùng posterior mặc định.

---

## Roadmap còn lại

- [ ] Điều tra `exploitation_suspected` (RL có thể đang khai thác surrogate model)
- [ ] Viết script train PPO gốc (hiện chỉ có checkpoint, chưa có `train_ppo.py`)
- [ ] Phase 8 — pilot thực tế (smart plug đo energy thật thay proxy)
- [ ] Chốt dùng GRU hay LGBM v2 nếu muốn thử nghiệm tiếp env model

---

## License

MIT — xem [`LICENSE`](LICENSE).
