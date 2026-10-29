# Hướng dẫn đưa repo lên GitHub và làm việc hằng ngày

Hướng dẫn này dành cho người **chưa từng dùng Git**. Làm lần lượt từ Phần 1 đến Phần 6 một lần duy nhất; từ lần sau chỉ cần Phần 7.

---

## Phần 0 · Năm khái niệm cần biết

| Thuật ngữ | Nghĩa |
| --- | --- |
| **Git** | Phần mềm trên máy bạn, ghi lại lịch sử thay đổi của thư mục mã nguồn. |
| **GitHub** | Trang web lưu bản sao thư mục đó trên mạng để chia sẻ, sao lưu và mở trên Colab. |
| **Repository (repo)** | Một thư mục dự án được Git theo dõi. |
| **Commit** | Một “ảnh chụp” có ghi chú của toàn bộ thay đổi tại một thời điểm. Có thể quay lại bất kỳ commit nào. |
| **Push / Pull** | Push = đẩy commit từ máy lên GitHub. Pull = kéo thay đổi mới từ GitHub về máy (hoặc về Colab). |

Quy trình luôn là: **sửa tệp → `git add` → `git commit` → `git push`**.

---

## Phần 1 · Chuẩn bị (một lần)

1. **Tạo tài khoản GitHub** tại https://github.com/signup. Nên dùng email trường (đủ điều kiện nhận GitHub Student Developer Pack).
2. **Cài Git**
   * Windows: tải tại https://git-scm.com/download/win, cài với các lựa chọn mặc định. Bộ cài kèm *Git Bash* (cửa sổ dòng lệnh) và *Git Credential Manager* (tự mở trình duyệt để đăng nhập GitHub khi push).
   * macOS: mở Terminal, gõ `git --version`; nếu chưa có, máy sẽ đề nghị cài Command Line Tools.
3. Kiểm tra: mở Git Bash (Windows) hoặc Terminal (macOS) và gõ:
   ```bash
   git --version
   ```
   Thấy dòng `git version 2.x.x` là được.

> Không quen dòng lệnh? Có thể dùng **GitHub Desktop** (https://desktop.github.com), xem Cách B ở Phần 4.

---

## Phần 2 · Khai báo tên và email (một lần)

```bash
git config --global user.name "Nguyen Van A"
git config --global user.email "email_dang_ky_github@example.com"
git config --global init.defaultBranch main
```

Email nên trùng với email đăng ký GitHub để commit hiện đúng tên bạn.

---

## Phần 3 · Tạo repo trống trên GitHub

1. Đăng nhập GitHub → bấm dấu **+** góc trên phải → **New repository**.
2. **Repository name**: `m5-calendar-forecasting`
3. **Description**: `Calendar-aware daily retail demand forecasting: LightGBM vs LSTM on M5`
4. Chọn **Private** (khuyến nghị cho đến khi nộp bài hoặc công bố; có thể đổi sang Public sau ở Settings).
5. **KHÔNG** tích “Add a README”, “.gitignore” hay “license” — repo đã có sẵn các tệp này; tích vào sẽ gây xung đột ở lần push đầu.
6. Bấm **Create repository**. GitHub hiện một trang hướng dẫn có đường dẫn dạng `https://github.com/<USERNAME>/m5-calendar-forecasting.git` — giữ trang này mở.

---

## Phần 4 · Đưa mã lên GitHub lần đầu

Trước hết, giải nén tệp `m5-calendar-forecasting.zip` vào một thư mục dễ tìm, ví dụ `D:\Projects\m5-calendar-forecasting` hoặc `~/Projects/m5-calendar-forecasting`.

### Cách A · Dòng lệnh (khuyến nghị)

Mở Git Bash / Terminal **trong thư mục đã giải nén** (Windows: chuột phải trong thư mục → *Open Git Bash here*), rồi chạy từng dòng:

```bash
# 1. Thay <USERNAME> trong các tệp bằng tên tài khoản GitHub của bạn (xem Phần 5) — làm trước khi commit

# 2. Khởi tạo Git trong thư mục
git init

# 3. Xem Git sẽ theo dõi những tệp nào
git status
```

Kiểm tra danh sách: **không được** có tệp `.csv`, `kaggle.json` hay thư mục `data/raw`. Nếu thấy, dừng lại và kiểm tra `.gitignore`.

```bash
# 4. Đưa mọi tệp vào "khu chờ"
git add .

# 5. Tạo commit đầu tiên
git commit -m "Khởi tạo repo: pipeline dự báo M5, kiểm thử chống rò rỉ, notebook Colab"

# 6. Đặt tên nhánh chính là main
git branch -M main

# 7. Nối thư mục với repo trên GitHub (thay <USERNAME>)
git remote add origin https://github.com/<USERNAME>/m5-calendar-forecasting.git

# 8. Đẩy lên GitHub
git push -u origin main
```

**Đăng nhập khi push**:
* Windows có Git Credential Manager: một cửa sổ trình duyệt mở ra → chọn *Sign in with your browser* → đăng nhập → *Authorize*. Lần sau không phải đăng nhập lại.
* Nếu được hỏi *Username/Password* trong dòng lệnh: GitHub **không nhận mật khẩu tài khoản**. Bạn cần một *Personal Access Token* (PAT):
  1. GitHub → ảnh đại diện → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
  2. Đặt tên (ví dụ `laptop-m5`), thời hạn 90 ngày; **Repository access** → *Only select repositories* → chọn `m5-calendar-forecasting`.
  3. **Permissions** → *Repository permissions* → **Contents: Read and write**.
  4. Bấm **Generate token**, sao chép chuỗi token (chỉ hiện một lần).
  5. Khi dòng lệnh hỏi: *Username* = tên tài khoản GitHub; *Password* = dán token.

Tải lại trang repo trên GitHub: bạn sẽ thấy toàn bộ tệp và README hiển thị bên dưới. Vào tab **Actions** sẽ thấy bộ kiểm thử tự chạy; dấu tích xanh nghĩa là mọi kiểm thử đạt.

### Cách B · GitHub Desktop (không cần dòng lệnh)

1. Cài GitHub Desktop, đăng nhập bằng tài khoản GitHub.
2. **File → Add local repository** → chọn thư mục đã giải nén → nó báo “not a Git repository” → bấm **create a repository** → **Create repository** (bỏ trống phần README/gitignore/license vì đã có).
3. Ô *Summary* ở góc dưới trái: nhập `Khởi tạo repo` → **Commit to main**.
4. Bấm **Publish repository** → bỏ chọn hoặc giữ *Keep this code private* → **Publish**.
   (Nếu đã tạo repo trống ở Phần 3, GitHub Desktop sẽ báo trùng tên; khi đó dùng *Repository → Repository settings → Remote* và dán đường dẫn repo, rồi bấm **Push origin**.)

### Cách C · Tải lên qua trang web (chỉ khi không cài được Git)

Trong repo trống → *uploading an existing file* → kéo thả toàn bộ nội dung thư mục (không kéo `data/raw`) → *Commit changes*. Cách này không giữ được thư mục rỗng và không thuận tiện cho các lần cập nhật sau.

---

## Phần 5 · Thay `<USERNAME>` bằng tên tài khoản của bạn

Các tệp có `<USERNAME>` cần sửa: `README.md`, `CITATION.cff`, và ô đầu tiên của ba notebook trong `notebooks/` (`GITHUB_USER = "<USERNAME>"`). Đồng thời sửa `[Họ tên tác giả]` trong `LICENSE` và `[Họ]`, `[Tên]` trong `CITATION.cff`.

Nếu bạn đã push trước khi sửa, sau khi sửa chạy:

```bash
git add .
git commit -m "Cập nhật tên tài khoản GitHub và thông tin tác giả"
git push
```

---

## Phần 6 · Mở repo trên Google Colab

1. Vào https://colab.research.google.com → hộp thoại *Open notebook* → tab **GitHub**.
2. Repo **Public**: dán `<USERNAME>/m5-calendar-forecasting` → chọn notebook.
   Repo **Private**: tích *Include private repos* → cho phép Colab truy cập GitHub.
3. Ô thiết lập trong notebook dùng `git clone` để tải mã về Colab. Với repo **Private**, `git clone` cần token:
   * Trong Colab, bấm biểu tượng chìa khóa (**Secrets**) ở thanh trái → *Add new secret* → tên `GITHUB_TOKEN`, giá trị = token PAT ở Phần 4 → bật *Notebook access*.
   * Thay dòng clone trong ô thiết lập bằng:
     ```python
     from google.colab import userdata
     token = userdata.get("GITHUB_TOKEN")
     !git clone https://{token}@github.com/{GITHUB_USER}/{REPO}.git /content/{REPO}
     ```
   * **Không bao giờ** dán token trực tiếp vào ô code rồi lưu notebook: token sẽ bị lưu cùng notebook.

Colab chỉ dùng để **chạy**; nên sửa mã trên máy (VS Code) hoặc ngay trên GitHub (mở tệp → biểu tượng bút chì), rồi trên Colab chạy lại ô thiết lập để `git pull` bản mới nhất.

---

## Phần 7 · Quy trình làm việc hằng ngày

```bash
git pull                     # 1. Lấy bản mới nhất trước khi sửa
# ... sửa mã / cấu hình ...
git status                   # 2. Xem tệp nào đã đổi
git diff                     # 3. Xem chi tiết thay đổi
git add src/m5cal/features.py configs/lgbm.yaml     # 4. Chọn tệp (hoặc git add .)
git commit -m "feat: thêm đặc trưng giá tương đối theo bộ phận"
git push                     # 5. Đẩy lên GitHub
```

**Quy ước ghi chú commit** (giúp đọc lại lịch sử khi viết báo cáo):

| Tiền tố | Dùng khi | Ví dụ |
| --- | --- | --- |
| `feat:` | thêm chức năng | `feat: thêm mô phỏng tồn kho theo loại ngày` |
| `fix:` | sửa lỗi | `fix: sửa rò rỉ chuẩn hóa giá trong SeqBuilder` |
| `exp:` | thay đổi cấu hình thí nghiệm | `exp: tăng input_len LSTM lên 112` |
| `docs:` | tài liệu | `docs: cập nhật nhật ký thí nghiệm tuần 3` |
| `test:` | kiểm thử | `test: thêm kiểm thử cho price_mode last_known` |

Commit nhỏ, thường xuyên; mỗi commit một ý.

### 7b · Cập nhật lên phiên bản 0.2.0 (đổi tên đề tài, thêm thí nghiệm bổ sung)

Phiên bản 0.2.0 đổi tên đề tài thành *“Giá trị của thông tin lịch trong dự báo nhu cầu bán lẻ theo ngày”*, thêm notebook thí nghiệm chính `notebooks/main/m5_calendar_value_lightgbm_lstm_v3.ipynb` và tệp `src/m5cal/supplementary.py`. **Không cần đổi tên repo**: tên `m5-calendar-forecasting` vẫn đúng với nội dung.

* **Nếu bạn chưa đưa repo lên GitHub:** giải nén bản zip mới và làm Phần 4 như bình thường.
* **Nếu đã đưa lên GitHub bản cũ:** giải nén bản zip mới vào một thư mục tạm, chép **đè** toàn bộ nội dung vào thư mục repo trên máy (giữ nguyên thư mục ẩn `.git`), rồi chạy:

```bash
cd m5-calendar-forecasting
git pull                                   # lấy thay đổi nếu bạn đã sửa trên GitHub
git status                                 # sẽ thấy README.md, CITATION.cff, docs/, src/m5cal/supplementary.py, notebooks/main/ ...
python -m pytest -q                        # nếu đã cài môi trường: phải không có lỗi
git add README.md CITATION.cff pyproject.toml src/m5cal/__init__.py docs/
git commit -m "docs: định vị lại đề tài theo giá trị thông tin lịch"
git add src/m5cal/supplementary.py tests/test_supplementary.py notebooks/main/
git commit -m "feat: thêm notebook v3 và thí nghiệm bổ sung (bootstrap, DM, permutation, tinh chỉnh LSTM)"
git push
git tag -a v0.2.0 -m "Đổi tên đề tài, thêm thí nghiệm bổ sung"
git push origin v0.2.0
```

* **Lưu kết quả chạy trên Colab về GitHub:** sau khi chạy notebook v3 trên Colab, chọn *File → Save a copy in GitHub*, chọn đúng repo và đường dẫn `notebooks/main/...`, ghi chú commit kiểu `exp: chạy mục 16 trên T4`. Trước khi lưu, kiểm tra notebook không chứa token hoặc nội dung `kaggle.json`.
* **Nếu vẫn muốn đổi tên repo:** GitHub → repo → *Settings* → *Repository name* → *Rename*. Sau đó trên máy chạy `git remote set-url origin https://github.com/<USERNAME>/<TÊN-MỚI>.git` và sửa đường dẫn trong `README.md`, `CITATION.cff`.

### 7c · Cập nhật lên phiên bản 0.3.0 (thiết kế 4 mô hình × 3 nhóm yếu tố lịch)

Làm giống mục 7b (chép đè bản zip mới vào thư mục repo, giữ `.git`), rồi:

```bash
git add README.md CITATION.cff pyproject.toml requirements.txt src/m5cal/__init__.py docs/
git commit -m "docs: đổi tên đề tài, thiết kế 3 nhóm yếu tố lịch × 4 họ mô hình"
git add src/m5cal/calendar_groups.py src/m5cal/arima_exog.py src/m5cal/nbeats_exog.py src/m5cal/group_effects.py tests/test_calendar_groups.py
git commit -m "feat: nhóm yếu tố lịch, ARIMAX, NBEATSx, bảng tác động theo nhóm"
git add notebooks/main/m5_calendar_impact_4models_v4.ipynb
git commit -m "exp: notebook v4 cho thí nghiệm chính"
git push
git tag -a v0.3.0 -m "Thiết kế 4 mô hình × 3 nhóm yếu tố lịch"
git push origin v0.3.0
```

---

## Phần 8 · Nhánh (branch) cho thay đổi lớn

Khi thử một ý tưởng có thể hỏng (ví dụ đổi kiến trúc LSTM), làm trên nhánh riêng để `main` luôn chạy được:

```bash
git checkout -b exp/lstm-attention      # tạo và chuyển sang nhánh mới
# ... sửa, commit nhiều lần ...
git push -u origin exp/lstm-attention   # đẩy nhánh lên GitHub
```

Trên GitHub sẽ hiện nút **Compare & pull request** → tạo Pull Request → xem tab *Checks* (kiểm thử phải xanh) → **Merge pull request** → về máy chạy:

```bash
git checkout main
git pull
```

---

## Phần 9 · Đánh dấu phiên bản và lấy DOI cho bài báo

Gắn thẻ (tag) cho các mốc quan trọng để biết kết quả nào ứng với mã nào:

```bash
git tag -a v0.1-proposal -m "Phiên bản nộp báo cáo đề xuất"
git push origin --tags
```

Để có DOI trích dẫn được trong mục *Data and code availability* của bài báo:

1. Chuyển repo sang **Public** (Settings → General → Danger Zone → Change visibility).
2. Đăng nhập https://zenodo.org bằng tài khoản GitHub → **GitHub** (trong menu tài khoản) → bật công tắc cho `m5-calendar-forecasting`.
3. Trên GitHub: **Releases → Draft a new release** → chọn tag (ví dụ `v1.0-submission`) → **Publish release**.
4. Zenodo tự lưu trữ bản phát hành và cấp DOI; dán DOI vào `CITATION.cff` và bài báo.

---

## Phần 10 · Lỗi thường gặp

| Thông báo | Nguyên nhân | Cách xử lý |
| --- | --- | --- |
| `! [rejected] main -> main (fetch first)` | GitHub có commit mới mà máy chưa có (ví dụ bạn sửa trên web) | `git pull --rebase` rồi `git push` |
| `remote: Support for password authentication was removed` | Dùng mật khẩu thay vì token | Tạo PAT (Phần 4) và dán vào ô Password |
| `fatal: remote origin already exists` | Đã chạy `git remote add` trước đó | `git remote set-url origin https://github.com/<USERNAME>/m5-calendar-forecasting.git` |
| `error: failed to push some refs` khi tạo repo có README | Repo trên GitHub không trống | `git pull origin main --allow-unrelated-histories`, sửa xung đột nếu có, rồi `git push` |
| `File ... exceeds GitHub's file size limit of 100 MB` | Lỡ commit dữ liệu hoặc mô hình | `git rm --cached <tệp>`, kiểm tra `.gitignore`, `git commit --amend`, rồi push lại |
| Lỡ commit `kaggle.json` hoặc token | Lộ thông tin bí mật | Vào Kaggle/GitHub **thu hồi khóa ngay** và tạo khóa mới; `git rm --cached kaggle.json`; xóa khỏi lịch sử bằng công cụ như `git filter-repo` |
| `warning: LF will be replaced by CRLF` | Khác kiểu xuống dòng Windows/Linux | Có thể bỏ qua |
| Merge conflict | Hai nơi sửa cùng một dòng | Mở tệp, tìm các dấu `<<<<<<<`, `=======`, `>>>>>>>`, giữ phần đúng, xóa các dấu, rồi `git add` và `git commit` |

---

## Phần 11 · Bảng lệnh tóm tắt

| Việc | Lệnh |
| --- | --- |
| Xem trạng thái | `git status` |
| Xem thay đổi chưa commit | `git diff` |
| Lịch sử commit | `git log --oneline --graph` |
| Lấy bản mới | `git pull` |
| Commit mọi thay đổi | `git add . && git commit -m "..."` |
| Đẩy lên | `git push` |
| Bỏ thay đổi chưa commit của một tệp | `git restore <tệp>` |
| Quay lại xem mã ở một commit cũ | `git checkout <mã-commit>` (quay về: `git checkout main`) |
| Tạo nhánh | `git checkout -b <tên>` |
| Gắn thẻ phiên bản | `git tag -a v1.0 -m "..."` rồi `git push origin --tags` |
