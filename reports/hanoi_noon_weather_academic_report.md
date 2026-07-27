---
title: "Báo cáo học thuật: Dự báo thời tiết Hà Nội lúc 12:00 ngày kế tiếp bằng Machine Learning"
author: "ML Final Project"
date: "2026-06-03"
lang: vi
---

# Tóm tắt

Nghiên cứu này xây dựng một hệ thống học máy nhằm dự báo thời tiết Hà Nội lúc 12:00 trưa ngày kế tiếp từ dữ liệu lịch sử Open-Meteo. Hai bài toán chính được xem xét là dự báo nhiệt độ dưới dạng hồi quy và dự báo mưa dưới dạng phân loại nhị phân. Dữ liệu ban đầu gồm 52.608 quan sát theo giờ trong giai đoạn 2020-2025; sau tiền xử lý chỉ giữ các bản ghi lúc 12:00, tập dữ liệu còn 2.192 quan sát và 2.191 mẫu học có nhãn ngày kế tiếp.

Hệ thống được thiết kế theo quy trình học máy có kiểm soát: tiền xử lý dữ liệu, tạo đặc trưng thời gian, đặc trưng trễ, đặc trưng rolling không rò rỉ dữ liệu, chia train/validation/test theo thời gian, huấn luyện mô hình, lựa chọn mô hình bằng validation set và đánh giá cuối trên test set. Kết quả cho thấy Linear Regression đạt hiệu quả tốt nhất cho dự báo nhiệt độ với RMSE test 2,319°C. Đối với dự báo mưa, SVM được chọn theo validation F1, đạt F1 test 0,673. Các mô hình học máy vượt baseline, đặc biệt trong bài toán mưa, nơi baseline majority class có F1 bằng 0 do không phát hiện được lớp mưa.

# 1. Giới thiệu

Dự báo thời tiết tại một thời điểm cụ thể trong ngày có ý nghĩa thực tiễn đối với đi lại, học tập, lao động ngoài trời và lập kế hoạch sinh hoạt. Trong nghiên cứu này, thời điểm 12:00 trưa được lựa chọn vì đây là giai đoạn hoạt động ban ngày rõ rệt, đồng thời các biến như nhiệt độ, bức xạ mặt trời, độ ẩm và mây có ảnh hưởng trực tiếp đến cảm nhận thời tiết.

Nội dung nghiên cứu gồm hai nhiệm vụ học máy:

- Dự báo nhiệt độ lúc 12:00 ngày kế tiếp, là bài toán hồi quy.
- Dự báo có mưa hay không lúc 12:00 ngày kế tiếp, là bài toán phân loại nhị phân.

Mức thời tiết như `cool`, `normal`, `hot` không được huấn luyện bằng mô hình riêng mà được suy ra từ nhiệt độ dự báo. Cách thiết kế này giúp tránh biến bài toán báo cáo thành một nhiệm vụ học máy không cần thiết.

# 2. Dữ liệu và phương pháp

## 2.1. Dữ liệu

Dữ liệu được thu thập từ Open-Meteo cho khu vực Hà Nội, gồm các biến khí tượng như nhiệt độ, độ ẩm tương đối, lượng mưa, mây che phủ, áp suất mực nước biển, tốc độ gió và bức xạ sóng ngắn. Sau khi lọc chỉ lấy thời điểm 12:00, dữ liệu có đặc điểm như sau:

| Thuộc tính | Giá trị |
|---|---:|
| Số bản ghi hourly ban đầu | 52.608 |
| Số bản ghi 12:00 sau tiền xử lý | 2.192 |
| Số mẫu supervised learning | 2.191 |
| Khoảng thời gian dữ liệu gốc | 2020-01-01 đến 2025-12-31 |
| Khoảng thời gian bản ghi 12:00 | 2020-01-01 đến 2025-12-31 |
| Tỷ lệ mưa ngày kế tiếp | 0,390 |

Dữ liệu không có giá trị thiếu ở các cột đầu vào sau khi đọc và chuẩn hóa. Điều này làm giảm rủi ro sai lệch do nội suy hoặc loại bỏ quá nhiều mẫu.

## 2.2. Thống kê mô tả

| Biến | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| temperature | 26,906 | 5,497 | 8,900 | 27,800 | 38,000 |
| humidity | 67,167 | 12,306 | 22,000 | 68,000 | 96,000 |
| precipitation | 0,255 | 0,957 | 0,000 | 0,000 | 13,800 |
| cloud_cover | 77,693 | 33,656 | 0,000 | 99,000 | 100,000 |
| pressure_msl | 1011,833 | 7,343 | 990,500 | 1011,300 | 1032,300 |
| wind_speed_10m | 9,766 | 5,027 | 0,000 | 9,300 | 38,000 |
| shortwave_radiation | 547,390 | 235,568 | 59,000 | 568,500 | 949,000 |

Nhiệt độ trung bình lúc 12:00 là 26,906°C, với độ lệch chuẩn 5,497°C. Điều này phản ánh biến thiên mùa vụ tương đối rõ nhưng không quá cực đoan. Lượng mưa có median bằng 0,000 mm, nghĩa là ít nhất một nửa số ngày không có mưa tại thời điểm 12:00. Tuy nhiên, độ lệch chuẩn lượng mưa lớn hơn mean, cho thấy phân phối lượng mưa lệch phải: đa số ngày không mưa hoặc mưa rất ít, trong khi một số ngày có mưa lớn. Đây là một nguyên nhân khiến bài toán phân loại mưa khó hơn bài toán hồi quy nhiệt độ.

Cloud cover có median 99%, cho thấy nhiều quan sát buổi trưa có độ che phủ mây cao. Shortwave radiation dao động rộng từ 59 đến 949, vì vậy biến này được đưa vào thí nghiệm A/B để kiểm tra liệu nó có bổ sung thông tin cho mô hình hay không.

## 2.3. Tiền xử lý dữ liệu

Quy trình tiền xử lý gồm các bước:

1. Phát hiện và bỏ các dòng metadata của Open-Meteo.
2. Chuẩn hóa tên cột.
3. Chuyển cột thời gian sang kiểu datetime.
4. Sắp xếp dữ liệu theo thời gian và loại bỏ timestamp trùng lặp.
5. Chỉ giữ các quan sát lúc 12:00.
6. Loại bỏ các cột `surface_pressure`, `wind_direction_10m` và `rain`.
7. Điền missing numeric bằng median nếu có.
8. Loại bỏ các dòng không có target sau khi tạo nhãn ngày kế tiếp.

Đặc biệt, cột `rain` bị loại bỏ khỏi tập đặc trưng để tránh trùng lặp thông tin với `precipitation` và hạn chế rủi ro rò rỉ dữ liệu trong bài toán phân loại mưa.

## 2.4. Tạo đặc trưng

Các đặc trưng được tạo theo ba nhóm chính:

- Đặc trưng thời gian: `month`, `day_of_year`, `season`.
- Đặc trưng trễ: lag 1, 3, 7 ngày cho nhiệt độ, độ ẩm và lượng mưa.
- Đặc trưng rolling: trung bình trượt nhiệt độ, độ ẩm và tổng lượng mưa trên cửa sổ 3 và 7 ngày.

Các đặc trưng rolling được tính bằng `shift(1)` trước khi `rolling()`. Đây là điểm quan trọng về phương pháp luận vì nó bảo đảm mô hình chỉ sử dụng thông tin quá khứ, không dùng dữ liệu của ngày cần dự báo.

Target được tạo như sau:

- `target_temperature = temperature.shift(-1)`
- `target_precipitation = precipitation.shift(-1)`
- `target_rain = target_precipitation > 0`

# 3. Thiết kế thực nghiệm

## 3.1. Chia tập dữ liệu

Dữ liệu được chia theo thời gian để phù hợp với bản chất chuỗi thời gian của thời tiết:

| Tập dữ liệu | Giai đoạn | Số mẫu | Tỷ lệ mưa |
|---|---|---:|---:|
| Training | 2020-2023 | 1.461 | 0,392 |
| Validation | 2024 | 366 | 0,380 |
| Test | 2025 | 364 | 0,393 |

Mô hình chỉ được huấn luyện trên training set. Validation set được dùng để lựa chọn mô hình và so sánh thí nghiệm. Test set chỉ được dùng để đánh giá cuối cùng. Cách chia này tránh việc thông tin từ tương lai ảnh hưởng đến quá trình huấn luyện.

## 3.2. Baseline

Hai baseline được sử dụng làm mốc so sánh:

- Temperature baseline: dự báo nhiệt độ ngày mai bằng nhiệt độ hôm nay. Đây là persistence baseline, phù hợp với đặc tính liên tục của nhiệt độ.
- Rain baseline: luôn dự báo lớp phổ biến nhất trong training set. Đây là majority class baseline, tương đương với `DummyClassifier(strategy="most_frequent")`.

Baseline không phải là mô hình học máy phức tạp. Chúng đóng vai trò kiểm tra xem các mô hình học máy có thực sự tạo ra giá trị vượt qua cách dự báo đơn giản hay không.

## 3.3. Mô hình và lựa chọn tham số

Đối với hồi quy nhiệt độ, nghiên cứu sử dụng:

- Linear Regression.
- KNN Regressor.
- XGBoost Regressor.

Đối với phân loại mưa, nghiên cứu sử dụng:

- Logistic Regression với `class_weight='balanced'`.
- SVM với `class_weight='balanced'`.
- XGBoost Classifier với `scale_pos_weight` tự động.

Quá trình tinh chỉnh tham số sử dụng `TimeSeriesSplit(n_splits=5)` và `RandomizedSearchCV` trên training set. Không sử dụng KFold ngẫu nhiên vì dữ liệu có thứ tự thời gian.

# 4. Trực quan hóa dữ liệu

## 4.1. Xu hướng nhiệt độ và phân phối

![Xu hướng nhiệt độ lúc 12:00](figures/temperature_trend.png)

Biểu đồ xu hướng nhiệt độ cho thấy chu kỳ mùa vụ rõ rệt qua các năm. Nhiệt độ tăng trong các tháng nóng và giảm trong các giai đoạn lạnh. Đây là cơ sở để thêm các đặc trưng thời gian như `month`, `season` và `day_of_year`.

![Phân phối nhiệt độ](figures/temperature_histogram.png)

Phân phối nhiệt độ cho thấy phần lớn quan sát nằm quanh vùng nhiệt độ trung bình đến nóng, phù hợp với khí hậu Hà Nội.

## 4.2. Correlation heatmap và PCA

![Correlation heatmap](figures/correlation_heatmap.png)

Correlation heatmap phản ánh quan hệ tuyến tính giữa các biến. Các đặc trưng nhiệt độ hiện tại, lag nhiệt độ và rolling temperature có xu hướng tương quan cao. Điều này hợp lý vì nhiệt độ có tính liên tục theo ngày. Tuy nhiên, tương quan cao cũng cho thấy khi diễn giải hệ số Linear Regression cần thận trọng do khả năng đa cộng tuyến.

![PCA visualization](figures/pca_visualization.png)

PCA giảm không gian đặc trưng nhiều chiều xuống hai thành phần chính PC1 và PC2. Các giá trị trên trục PCA không còn là đơn vị khí tượng gốc như °C hay mm, mà là tọa độ tổng hợp sau phép biến đổi tuyến tính. Nếu các điểm mưa và không mưa chồng lấn nhiều, điều đó cho thấy bài toán phân loại mưa khó tách chỉ bằng hai chiều biểu diễn chính.

# 5. Kết quả

## 5.1. Kết quả hồi quy nhiệt độ

Kết quả trên validation set:

| Mô hình | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 1,811 | 2,375 | 0,814 |
| KNN Regressor | 2,214 | 2,761 | 0,749 |
| XGBoost Regressor | 1,893 | 2,439 | 0,804 |

Kết quả trên test set:

| Mô hình | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 1,674 | 2,319 | 0,808 |
| KNN Regressor | 2,148 | 2,821 | 0,716 |
| XGBoost Regressor | 1,830 | 2,432 | 0,789 |

Linear Regression được chọn vì có RMSE thấp nhất trên validation set. Trên test set, mô hình này tiếp tục đạt kết quả tốt nhất với RMSE 2,319°C. Kết quả này cho thấy quan hệ giữa nhiệt độ hiện tại/quá khứ và nhiệt độ ngày kế tiếp có tính tuyến tính tương đối mạnh.

![So sánh mô hình hồi quy](figures/regression_model_comparison.png)

So với baseline persistence, Linear Regression cải thiện RMSE từ 2,516 xuống 2,319 trên test set. Mức cải thiện không quá lớn vì baseline đã là một mốc mạnh đối với nhiệt độ, nhưng sự cải thiện này vẫn cho thấy mô hình học máy khai thác thêm được thông tin từ các đặc trưng thời gian, lag và rolling.

## 5.2. Phân tích dự báo nhiệt độ trên validation và test

![Validation actual vs predicted temperature](figures/validation_actual_vs_predicted_temperature.png)

![Test actual vs predicted temperature](figures/test_actual_vs_predicted_temperature.png)

Biểu đồ actual vs predicted cho thấy các điểm dự báo tập trung quanh đường chéo lý tưởng. Các điểm càng gần đường chéo thì dự báo càng chính xác. Những điểm lệch xa phản ánh các ngày có biến động thời tiết mạnh hoặc trạng thái khí tượng khó học từ đặc trưng hiện có.

![Temperature tolerance accuracy](figures/temperature_tolerance_accuracy.png)

Đối với hồi quy, khái niệm đúng/sai tuyệt đối không phù hợp. Do đó, nghiên cứu báo cáo tỷ lệ dự báo nằm trong các ngưỡng sai số:

| Tập dữ liệu | MAE | RMSE | Trong ±1°C | Trong ±2°C | Trong ±3°C |
|---|---:|---:|---:|---:|---:|
| Validation | 1,811 | 2,375 | 0,388 | 0,653 | 0,806 |
| Test | 1,674 | 2,319 | 0,420 | 0,692 | 0,849 |

Trên test set, khoảng 69,2% dự báo nằm trong sai số ±2°C và 84,9% nằm trong ±3°C. Đây là kết quả tương đối hợp lý đối với một hệ thống sử dụng mô hình truyền thống và dữ liệu quan sát lịch sử.

## 5.3. Kết quả phân loại mưa

Kết quả trên validation set:

| Mô hình | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,653 | 0,532 | 0,712 | 0,609 | 0,742 |
| SVM | 0,680 | 0,560 | 0,741 | 0,638 | 0,758 |
| XGBoost Classifier | 0,686 | 0,574 | 0,669 | 0,618 | 0,751 |

Kết quả trên test set:

| Mô hình | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,720 | 0,612 | 0,783 | 0,687 | 0,785 |
| SVM | 0,717 | 0,616 | 0,741 | 0,673 | 0,774 |
| XGBoost Classifier | 0,698 | 0,611 | 0,636 | 0,623 | 0,767 |

SVM được chọn vì có F1 cao nhất trên validation set. Trên test set, SVM đạt F1 0,673 và accuracy 0,717. Đáng chú ý, Logistic Regression có F1 test cao hơn SVM, đạt 0,687. Điều này cho thấy lựa chọn mô hình dựa trên một năm validation có thể nhạy với biến động phân phối giữa các năm.

![So sánh mô hình phân loại](figures/classification_model_comparison.png)

## 5.4. Confusion matrix và tỷ lệ đúng/sai

![Validation confusion matrix](figures/validation_confusion_matrix_rain.png)

![Test confusion matrix](figures/test_confusion_matrix_rain.png)

Confusion matrix cho biết mô hình dự báo đúng/sai như thế nào ở từng lớp mưa và không mưa. Với SVM, trên test set có 261 dự báo đúng và 103 dự báo sai. Recall test đạt 0,741, nghĩa là mô hình phát hiện được khoảng 74,1% số ngày mưa thực tế. Precision test đạt 0,616, nghĩa là trong các ngày mô hình dự báo mưa, khoảng 61,6% là mưa thật.

![Rain correct wrong rate](figures/rain_correct_wrong_rate.png)

Baseline majority class có F1 bằng 0 vì chỉ dự báo lớp không mưa, do đó không phát hiện được ngày mưa nào. Các mô hình học máy cải thiện rõ rệt so với baseline ở khả năng nhận diện lớp mưa.

# 6. Thí nghiệm A/B với shortwave radiation

Thí nghiệm A/B kiểm tra ảnh hưởng của biến `shortwave_radiation`.

| Thí nghiệm | Dùng shortwave | Best regression | Validation RMSE | Test RMSE | Best rain | Validation F1 | Test F1 |
|---|---:|---|---:|---:|---|---:|---:|
| A_without_shortwave_radiation | False | Linear Regression | 2,374 | 2,317 | SVM | 0,652 | 0,660 |
| B_with_shortwave_radiation | True | Linear Regression | 2,375 | 2,319 | SVM | 0,638 | 0,673 |

Đối với hồi quy nhiệt độ, thêm `shortwave_radiation` không cải thiện rõ rệt; RMSE gần như tương đương và hơi kém hơn. Đối với phân loại mưa, không dùng shortwave tốt hơn trên validation nhưng dùng shortwave tốt hơn trên test. Kết quả này cho thấy biến shortwave có thể chứa tín hiệu khí tượng, nhưng tác động chưa ổn định giữa các năm.

![So sánh A/B](figures/experiment_ab_comparison.png)

# 7. Giải thích mô hình và feature importance

Feature importance của XGBoost Regressor cho thấy các đặc trưng quan trọng nhất gồm:

| Feature | Importance |
|---|---:|
| temperature | 0,484 |
| season | 0,176 |
| day_of_year | 0,032 |
| pressure_msl | 0,027 |
| temp_lag7 | 0,021 |
| temp_roll_mean_7 | 0,020 |
| temp_lag1 | 0,018 |
| wind_speed_10m | 0,018 |
| rain_roll_sum_7 | 0,016 |
| month | 0,016 |

![Feature importance](figures/feature_importance.png)

Kết quả này phù hợp với trực giác khí tượng. Nhiệt độ hiện tại là biến quan trọng nhất vì nhiệt độ có tính liên tục mạnh. Các biến mùa vụ như `season`, `month`, `day_of_year` giúp mô hình nhận biết chu kỳ khí hậu. Các biến lag và rolling phản ánh trạng thái thời tiết trong quá khứ gần.

Cần nhấn mạnh rằng feature importance không dùng để chọn mô hình tốt nhất. Mô hình tốt nhất được chọn bằng metric trên validation set, còn feature importance chỉ dùng để diễn giải cách một mô hình cụ thể sử dụng đặc trưng.

# 8. Thảo luận

Kết quả nghiên cứu cho thấy bài toán dự báo nhiệt độ dễ hơn bài toán dự báo mưa. Đối với nhiệt độ, mô hình tuyến tính đơn giản đã đạt hiệu quả tốt nhất, chứng tỏ quan hệ giữa các đặc trưng lịch sử và nhiệt độ ngày kế tiếp tương đối ổn định. Điều này cũng cho thấy không phải lúc nào mô hình phức tạp như XGBoost cũng vượt trội; trong bối cảnh dữ liệu có cấu trúc tuyến tính mạnh, mô hình đơn giản có thể tổng quát hóa tốt hơn.

Đối với mưa, kết quả có tính bất định cao hơn. SVM thắng trên validation, nhưng Logistic Regression lại đạt F1 cao hơn trên test. Đây là dấu hiệu cho thấy phân phối mưa giữa các năm có thể thay đổi, và việc chỉ dùng một năm validation có thể chưa đủ để kết luận tuyệt đối. Tuy vậy, tất cả mô hình học máy đều vượt baseline majority class về F1, chứng minh rằng các đặc trưng khí tượng có chứa tín hiệu hữu ích cho việc nhận diện mưa.

Thí nghiệm A/B cho thấy `shortwave_radiation` chưa đem lại cải thiện nhất quán. Với nhiệt độ, biến này không giúp giảm RMSE rõ rệt. Với mưa, hiệu quả khác nhau giữa validation và test. Do đó, biến này nên được giữ như một ứng viên đặc trưng cần kiểm chứng thêm, thay vì kết luận chắc chắn rằng nó cải thiện mô hình.

# 9. Hạn chế

Nghiên cứu có một số hạn chế:

- Dữ liệu chỉ đại diện cho một vị trí tại Hà Nội, chưa đánh giá khả năng tổng quát trên nhiều khu vực.
- Dữ liệu đầu vào là quan sát lịch sử, chưa sử dụng forecast variables từ mô hình khí tượng số.
- Validation set chỉ gồm một năm, nên việc chọn mô hình có thể nhạy với biến động khí hậu từng năm.
- Bài toán mưa có tính bất định cao, có thể cần thêm dữ liệu radar, vệ tinh, hoặc biến khí tượng chuyên sâu.
- Các mô hình hiện tại chủ yếu là mô hình truyền thống, chưa khai thác đầy đủ cấu trúc chuỗi thời gian dài hạn.

# 10. Kết luận và hướng phát triển

Nghiên cứu đã xây dựng thành công một pipeline học máy có khả năng dự báo thời tiết Hà Nội lúc 12:00 ngày kế tiếp. Với bài toán nhiệt độ, Linear Regression đạt kết quả tốt nhất và cải thiện so với baseline persistence. Với bài toán mưa, SVM được chọn theo validation F1 và đạt hiệu quả tốt hơn baseline majority class, dù kết quả còn dao động giữa validation và test.

Kết quả tổng thể cho thấy dữ liệu lịch sử lúc 12:00 có đủ tín hiệu để dự báo nhiệt độ ngày kế tiếp ở mức chấp nhận được. Tuy nhiên, dự báo mưa vẫn là bài toán khó hơn và cần thêm đặc trưng khí tượng để cải thiện độ tin cậy.

Các hướng phát triển tiếp theo gồm:

- Mở rộng dữ liệu sang nhiều vị trí trong Hà Nội và miền Bắc.
- Bổ sung forecast variables từ các mô hình khí tượng số.
- Thử nghiệm mô hình chuỗi thời gian như LSTM, Temporal CNN hoặc Transformer.
- Hiệu chỉnh xác suất mưa thay vì chỉ dự báo nhãn nhị phân.
- Xây dựng dashboard hoặc API để triển khai hệ thống dự báo.
