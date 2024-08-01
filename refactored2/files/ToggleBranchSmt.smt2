(declare-fun age0 () Int)

(declare-fun workclass0 () Int)

(declare-fun fnlwgt0 () Int)

(declare-fun education0 () Int)

(declare-fun martial_status0 () Int)

(declare-fun occupation0 () Int)

(declare-fun relationship0 () Int)

(declare-fun race0 () Int)

(declare-fun sex0 () Int)

(declare-fun capital_gain0 () Int)

(declare-fun capital_loss0 () Int)

(declare-fun hours_per_week0 () Int)

(declare-fun native_country0 () Int)

; 0th element
(declare-fun age1 () Int)

(declare-fun workclass1 () Int)

(declare-fun fnlwgt1 () Int)

(declare-fun education1 () Int)

(declare-fun martial_status1 () Int)

(declare-fun occupation1 () Int)

(declare-fun relationship1 () Int)

(declare-fun race1 () Int)

(declare-fun sex1 () Int)

(declare-fun capital_gain1 () Int)

(declare-fun capital_loss1 () Int)

(declare-fun hours_per_week1 () Int)

(declare-fun native_country1 () Int)

; 1th element
(declare-fun Class0 () Int)
(declare-fun Class1 () Int)
(define-fun absoluteInt ((x Int)) Int
(ite (>= x 0) x (- x)))
(define-fun absoluteReal ((x Real)) Real
(ite (>= x 0) x (- x)))

;-----------0-----------number instance--------------
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (<= hours_per_week0 11.0) (<= capital_gain0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (<= hours_per_week0 11.0) (> capital_gain0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (<= hours_per_week0 21.0) (<= occupation0 2.0) (<= hours_per_week0 17.0) (<= workclass0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (<= hours_per_week0 21.0) (<= occupation0 2.0) (<= hours_per_week0 17.0) (> workclass0 0.0) (<= age0 5.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (<= hours_per_week0 21.0) (<= occupation0 2.0) (<= hours_per_week0 17.0) (> workclass0 0.0) (> age0 5.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (<= hours_per_week0 21.0) (<= occupation0 2.0) (> hours_per_week0 17.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (<= hours_per_week0 21.0) (> occupation0 2.0) (<= fnlwgt0 7.0) (<= workclass0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (<= hours_per_week0 21.0) (> occupation0 2.0) (<= fnlwgt0 7.0) (> workclass0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (<= hours_per_week0 21.0) (> occupation0 2.0) (> fnlwgt0 7.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (<= race0 0.0) (> hours_per_week0 21.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (<= occupation0 3.0) (> hours_per_week0 11.0) (> race0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (<= hours_per_week0 19.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (> hours_per_week0 19.0) (<= occupation0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (> hours_per_week0 19.0) (> occupation0 4.0) (<= hours_per_week0 24.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (> hours_per_week0 19.0) (> occupation0 4.0) (> hours_per_week0 24.0) (<= occupation0 5.0) (<= education0 2.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (> hours_per_week0 19.0) (> occupation0 4.0) (> hours_per_week0 24.0) (<= occupation0 5.0) (> education0 2.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (> hours_per_week0 19.0) (> occupation0 4.0) (> hours_per_week0 24.0) (> occupation0 5.0) (<= age0 5.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (> hours_per_week0 19.0) (> occupation0 4.0) (> hours_per_week0 24.0) (> occupation0 5.0) (> age0 5.0) (<= hours_per_week0 28.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (<= workclass0 0.0) (> hours_per_week0 19.0) (> occupation0 4.0) (> hours_per_week0 24.0) (> occupation0 5.0) (> age0 5.0) (> hours_per_week0 28.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (<= capital_loss0 9.0) (> workclass0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (<= capital_gain0 0.0) (> capital_loss0 9.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (<= hours_per_week0 31.0) (> occupation0 3.0) (> capital_gain0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (<= occupation0 4.0) (<= native_country0 5.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (<= occupation0 4.0) (> native_country0 5.0) (<= native_country0 7.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (<= occupation0 4.0) (> native_country0 5.0) (> native_country0 7.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (<= hours_per_week0 39.0) (<= workclass0 0.0) (<= fnlwgt0 27.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (<= hours_per_week0 39.0) (<= workclass0 0.0) (> fnlwgt0 27.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (<= hours_per_week0 39.0) (> workclass0 0.0) (<= fnlwgt0 6.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (<= hours_per_week0 39.0) (> workclass0 0.0) (> fnlwgt0 6.0) (<= education0 3.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (<= hours_per_week0 39.0) (> workclass0 0.0) (> fnlwgt0 6.0) (> education0 3.0) (<= fnlwgt0 8.0) (<= hours_per_week0 36.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (<= hours_per_week0 39.0) (> workclass0 0.0) (> fnlwgt0 6.0) (> education0 3.0) (<= fnlwgt0 8.0) (> hours_per_week0 36.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (<= hours_per_week0 39.0) (> workclass0 0.0) (> fnlwgt0 6.0) (> education0 3.0) (> fnlwgt0 8.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (<= native_country0 3.0) (<= workclass0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (<= native_country0 3.0) (> workclass0 0.0) (<= occupation0 5.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (<= native_country0 3.0) (> workclass0 0.0) (> occupation0 5.0) (<= hours_per_week0 42.0) (<= workclass0 1.0) (<= fnlwgt0 3.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (<= native_country0 3.0) (> workclass0 0.0) (> occupation0 5.0) (<= hours_per_week0 42.0) (<= workclass0 1.0) (> fnlwgt0 3.0) (<= occupation0 6.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (<= native_country0 3.0) (> workclass0 0.0) (> occupation0 5.0) (<= hours_per_week0 42.0) (<= workclass0 1.0) (> fnlwgt0 3.0) (> occupation0 6.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (<= native_country0 3.0) (> workclass0 0.0) (> occupation0 5.0) (<= hours_per_week0 42.0) (> workclass0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (<= native_country0 3.0) (> workclass0 0.0) (> occupation0 5.0) (> hours_per_week0 42.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (> native_country0 3.0) (<= hours_per_week0 43.0) (<= occupation0 6.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (> native_country0 3.0) (<= hours_per_week0 43.0) (> occupation0 6.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (<= race0 0.0) (> occupation0 4.0) (> hours_per_week0 39.0) (> native_country0 3.0) (> hours_per_week0 43.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (<= occupation0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (<= race0 3.0) (<= native_country0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (<= race0 3.0) (> native_country0 4.0) (<= hours_per_week0 45.0) (<= capital_gain0 2.0) (<= fnlwgt0 14.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (<= race0 3.0) (> native_country0 4.0) (<= hours_per_week0 45.0) (<= capital_gain0 2.0) (> fnlwgt0 14.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (<= race0 3.0) (> native_country0 4.0) (<= hours_per_week0 45.0) (> capital_gain0 2.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (<= race0 3.0) (> native_country0 4.0) (> hours_per_week0 45.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (<= occupation0 2.0) (<= age0 3.0) (<= fnlwgt0 8.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (<= occupation0 2.0) (<= age0 3.0) (> fnlwgt0 8.0) (<= education0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (<= occupation0 2.0) (<= age0 3.0) (> fnlwgt0 8.0) (> education0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (<= occupation0 2.0) (> age0 3.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (> occupation0 2.0) (<= age0 2.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (> occupation0 2.0) (> age0 2.0) (<= education0 2.0) (<= hours_per_week0 45.0) (<= age0 5.0) (<= fnlwgt0 21.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (> occupation0 2.0) (> age0 2.0) (<= education0 2.0) (<= hours_per_week0 45.0) (<= age0 5.0) (> fnlwgt0 21.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (> occupation0 2.0) (> age0 2.0) (<= education0 2.0) (<= hours_per_week0 45.0) (> age0 5.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (> occupation0 2.0) (> age0 2.0) (<= education0 2.0) (> hours_per_week0 45.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (> occupation0 2.0) (> age0 2.0) (> education0 2.0) (<= sex0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (<= hours_per_week0 55.0) (> occupation0 2.0) (> age0 2.0) (> education0 2.0) (> sex0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (<= occupation0 5.0) (> occupation0 1.0) (> race0 3.0) (> hours_per_week0 55.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (> occupation0 5.0) (<= capital_gain0 1.0) (<= hours_per_week0 55.0) (<= capital_loss0 9.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (> occupation0 5.0) (<= capital_gain0 1.0) (<= hours_per_week0 55.0) (> capital_loss0 9.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (> occupation0 5.0) (<= capital_gain0 1.0) (> hours_per_week0 55.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (<= occupation0 7.0) (> hours_per_week0 31.0) (> race0 0.0) (> occupation0 5.0) (> capital_gain0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (<= hours_per_week0 35.0) (<= capital_gain0 1.0) (<= capital_loss0 8.0) (<= hours_per_week0 32.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (<= hours_per_week0 35.0) (<= capital_gain0 1.0) (<= capital_loss0 8.0) (> hours_per_week0 32.0) (<= fnlwgt0 6.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (<= hours_per_week0 35.0) (<= capital_gain0 1.0) (<= capital_loss0 8.0) (> hours_per_week0 32.0) (> fnlwgt0 6.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (<= hours_per_week0 35.0) (<= capital_gain0 1.0) (> capital_loss0 8.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (<= hours_per_week0 35.0) (> capital_gain0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (<= fnlwgt0 10.0) (<= native_country0 1.0) (<= age0 2.0) (<= fnlwgt0 5.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (<= fnlwgt0 10.0) (<= native_country0 1.0) (<= age0 2.0) (> fnlwgt0 5.0) (<= sex0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (<= fnlwgt0 10.0) (<= native_country0 1.0) (<= age0 2.0) (> fnlwgt0 5.0) (> sex0 0.0) (<= capital_gain0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (<= fnlwgt0 10.0) (<= native_country0 1.0) (<= age0 2.0) (> fnlwgt0 5.0) (> sex0 0.0) (> capital_gain0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (<= fnlwgt0 10.0) (<= native_country0 1.0) (> age0 2.0) (<= workclass0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (<= fnlwgt0 10.0) (<= native_country0 1.0) (> age0 2.0) (> workclass0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (<= fnlwgt0 10.0) (> native_country0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (<= hours_per_week0 41.0) (<= sex0 0.0) (<= fnlwgt0 15.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (<= hours_per_week0 41.0) (<= sex0 0.0) (> fnlwgt0 15.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (<= hours_per_week0 41.0) (> sex0 0.0) (<= age0 3.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (<= hours_per_week0 41.0) (> sex0 0.0) (> age0 3.0) (<= fnlwgt0 14.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (<= hours_per_week0 41.0) (> sex0 0.0) (> age0 3.0) (> fnlwgt0 14.0) (<= age0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (<= hours_per_week0 41.0) (> sex0 0.0) (> age0 3.0) (> fnlwgt0 14.0) (> age0 4.0) (<= fnlwgt0 24.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (<= hours_per_week0 41.0) (> sex0 0.0) (> age0 3.0) (> fnlwgt0 14.0) (> age0 4.0) (> fnlwgt0 24.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (<= race0 0.0) (> hours_per_week0 35.0) (> fnlwgt0 10.0) (> hours_per_week0 41.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (> race0 0.0) (<= capital_gain0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (<= occupation0 8.0) (> race0 0.0) (> capital_gain0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (<= hours_per_week0 44.0) (<= fnlwgt0 1.0) (<= workclass0 0.0) (<= occupation0 9.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (<= hours_per_week0 44.0) (<= fnlwgt0 1.0) (<= workclass0 0.0) (> occupation0 9.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (<= hours_per_week0 44.0) (<= fnlwgt0 1.0) (> workclass0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (<= hours_per_week0 44.0) (> fnlwgt0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (<= workclass0 0.0) (<= fnlwgt0 10.0) (<= education0 2.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (<= workclass0 0.0) (<= fnlwgt0 10.0) (> education0 2.0) (<= race0 0.0) (<= fnlwgt0 7.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (<= workclass0 0.0) (<= fnlwgt0 10.0) (> education0 2.0) (<= race0 0.0) (> fnlwgt0 7.0) (<= age0 3.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (<= workclass0 0.0) (<= fnlwgt0 10.0) (> education0 2.0) (<= race0 0.0) (> fnlwgt0 7.0) (> age0 3.0) (<= native_country0 2.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (<= workclass0 0.0) (<= fnlwgt0 10.0) (> education0 2.0) (<= race0 0.0) (> fnlwgt0 7.0) (> age0 3.0) (> native_country0 2.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (<= workclass0 0.0) (<= fnlwgt0 10.0) (> education0 2.0) (> race0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (<= workclass0 0.0) (> fnlwgt0 10.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (<= capital_loss0 9.0) (> hours_per_week0 44.0) (> workclass0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (<= capital_gain0 1.0) (> capital_loss0 9.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (<= hours_per_week0 49.0) (> occupation0 8.0) (> capital_gain0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (<= fnlwgt0 17.0) (<= workclass0 0.0) (<= education0 12.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (<= fnlwgt0 17.0) (<= workclass0 0.0) (> education0 12.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (<= fnlwgt0 17.0) (> workclass0 0.0) (<= fnlwgt0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (<= fnlwgt0 17.0) (> workclass0 0.0) (> fnlwgt0 4.0) (<= age0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (<= fnlwgt0 17.0) (> workclass0 0.0) (> fnlwgt0 4.0) (> age0 4.0) (<= fnlwgt0 6.0) (<= workclass0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (<= fnlwgt0 17.0) (> workclass0 0.0) (> fnlwgt0 4.0) (> age0 4.0) (<= fnlwgt0 6.0) (> workclass0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (<= fnlwgt0 17.0) (> workclass0 0.0) (> fnlwgt0 4.0) (> age0 4.0) (> fnlwgt0 6.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (<= hours_per_week0 50.0) (> fnlwgt0 17.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (<= native_country0 9.0) (> occupation0 7.0) (> hours_per_week0 49.0) (> hours_per_week0 50.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (<= native_country0 14.0) (<= occupation0 2.0) (<= workclass0 0.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (<= native_country0 14.0) (<= occupation0 2.0) (> workclass0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (<= native_country0 14.0) (> occupation0 2.0) (<= hours_per_week0 49.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (<= native_country0 14.0) (> occupation0 2.0) (> hours_per_week0 49.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (> native_country0 14.0) (<= occupation0 1.0) (<= capital_loss0 7.0) (<= native_country0 18.0) (<= age0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (> native_country0 14.0) (<= occupation0 1.0) (<= capital_loss0 7.0) (<= native_country0 18.0) (> age0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (> native_country0 14.0) (<= occupation0 1.0) (<= capital_loss0 7.0) (> native_country0 18.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (> native_country0 14.0) (<= occupation0 1.0) (> capital_loss0 7.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (<= capital_loss0 15.0) (> native_country0 14.0) (> occupation0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (<= hours_per_week0 52.0) (> capital_loss0 15.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (> hours_per_week0 52.0) (<= native_country0 29.0) (<= fnlwgt0 26.0) (<= race0 2.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (> hours_per_week0 52.0) (<= native_country0 29.0) (<= fnlwgt0 26.0) (> race0 2.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (> hours_per_week0 52.0) (<= native_country0 29.0) (> fnlwgt0 26.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (<= capital_gain0 1.0) (> hours_per_week0 52.0) (> native_country0 29.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (> capital_gain0 1.0) (<= native_country0 78.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (> capital_gain0 1.0) (> native_country0 78.0) (<= education0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (> capital_gain0 1.0) (> native_country0 78.0) (> education0 1.0) (<= fnlwgt0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (<= workclass0 2.0) (> native_country0 9.0) (> capital_gain0 1.0) (> native_country0 78.0) (> education0 1.0) (> fnlwgt0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (<= occupation0 1.0) (<= race0 0.0) (<= workclass0 4.0) (<= hours_per_week0 27.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (<= occupation0 1.0) (<= race0 0.0) (<= workclass0 4.0) (> hours_per_week0 27.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (<= occupation0 1.0) (<= race0 0.0) (> workclass0 4.0) (<= fnlwgt0 5.0) (<= education0 5.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (<= occupation0 1.0) (<= race0 0.0) (> workclass0 4.0) (<= fnlwgt0 5.0) (> education0 5.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (<= occupation0 1.0) (<= race0 0.0) (> workclass0 4.0) (> fnlwgt0 5.0) (<= age0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (<= occupation0 1.0) (<= race0 0.0) (> workclass0 4.0) (> fnlwgt0 5.0) (> age0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (<= occupation0 1.0) (> race0 0.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (<= workclass0 4.0) (<= native_country0 5.0) (<= hours_per_week0 35.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (<= workclass0 4.0) (<= native_country0 5.0) (> hours_per_week0 35.0) (<= age0 4.0) (<= fnlwgt0 3.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (<= workclass0 4.0) (<= native_country0 5.0) (> hours_per_week0 35.0) (<= age0 4.0) (> fnlwgt0 3.0) (<= education0 2.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (<= workclass0 4.0) (<= native_country0 5.0) (> hours_per_week0 35.0) (<= age0 4.0) (> fnlwgt0 3.0) (> education0 2.0) (<= workclass0 3.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (<= workclass0 4.0) (<= native_country0 5.0) (> hours_per_week0 35.0) (<= age0 4.0) (> fnlwgt0 3.0) (> education0 2.0) (> workclass0 3.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (<= workclass0 4.0) (<= native_country0 5.0) (> hours_per_week0 35.0) (> age0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (<= workclass0 4.0) (> native_country0 5.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (<= occupation0 3.0) (> workclass0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (<= hours_per_week0 47.0) (<= workclass0 3.0) (<= occupation0 4.0) (<= race0 1.0) (<= education0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (<= hours_per_week0 47.0) (<= workclass0 3.0) (<= occupation0 4.0) (<= race0 1.0) (> education0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (<= hours_per_week0 47.0) (<= workclass0 3.0) (<= occupation0 4.0) (> race0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (<= hours_per_week0 47.0) (<= workclass0 3.0) (> occupation0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (<= hours_per_week0 47.0) (> workclass0 3.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (<= workclass0 4.0) (<= fnlwgt0 16.0) (<= workclass0 3.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (<= workclass0 4.0) (<= fnlwgt0 16.0) (> workclass0 3.0) (<= age0 4.0) (<= occupation0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (<= workclass0 4.0) (<= fnlwgt0 16.0) (> workclass0 3.0) (<= age0 4.0) (> occupation0 4.0) (<= relationship0 1.0) (<= fnlwgt0 12.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (<= workclass0 4.0) (<= fnlwgt0 16.0) (> workclass0 3.0) (<= age0 4.0) (> occupation0 4.0) (<= relationship0 1.0) (> fnlwgt0 12.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (<= workclass0 4.0) (<= fnlwgt0 16.0) (> workclass0 3.0) (<= age0 4.0) (> occupation0 4.0) (> relationship0 1.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (<= workclass0 4.0) (<= fnlwgt0 16.0) (> workclass0 3.0) (> age0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (<= workclass0 4.0) (> fnlwgt0 16.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (<= capital_gain0 1.0) (> occupation0 3.0) (> hours_per_week0 47.0) (> workclass0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (<= hours_per_week0 51.0) (> capital_gain0 1.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (> hours_per_week0 51.0) (<= native_country0 2.0) (<= education0 11.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (> hours_per_week0 51.0) (<= native_country0 2.0) (> education0 11.0) (<= age0 4.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (> hours_per_week0 51.0) (<= native_country0 2.0) (> education0 11.0) (> age0 4.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (<= capital_loss0 6.0) (> occupation0 1.0) (> hours_per_week0 51.0) (> native_country0 2.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (<= occupation0 7.0) (> capital_loss0 6.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (<= capital_loss0 13.0) (<= capital_gain0 1.0) (<= hours_per_week0 58.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (<= capital_loss0 13.0) (<= capital_gain0 1.0) (> hours_per_week0 58.0) (<= workclass0 3.0) (<= race0 2.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (<= capital_loss0 13.0) (<= capital_gain0 1.0) (> hours_per_week0 58.0) (<= workclass0 3.0) (> race0 2.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (<= capital_loss0 13.0) (<= capital_gain0 1.0) (> hours_per_week0 58.0) (> workclass0 3.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (<= capital_loss0 13.0) (> capital_gain0 1.0) (<= workclass0 5.0) (<= age0 6.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (<= capital_loss0 13.0) (> capital_gain0 1.0) (<= workclass0 5.0) (> age0 6.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (<= capital_loss0 13.0) (> capital_gain0 1.0) (> workclass0 5.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (> capital_loss0 13.0) (<= workclass0 24.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (<= capital_gain0 6.0) (> occupation0 7.0) (> capital_loss0 13.0) (> workclass0 24.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (> capital_gain0 6.0) (<= capital_gain0 28.0) (<= workclass0 37.0) (<= native_country0 93.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (> capital_gain0 6.0) (<= capital_gain0 28.0) (<= workclass0 37.0) (> native_country0 93.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (> capital_gain0 6.0) (<= capital_gain0 28.0) (> workclass0 37.0) (<= occupation0 23.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (> capital_gain0 6.0) (<= capital_gain0 28.0) (> workclass0 37.0) (> occupation0 23.0) (<= capital_loss0 39.0) ) (= Class0 0)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (> capital_gain0 6.0) (<= capital_gain0 28.0) (> workclass0 37.0) (> occupation0 23.0) (> capital_loss0 39.0) ) (= Class0 1)))
(assert (=> (and (<= martial_status0 0.0) (> workclass0 2.0) (> capital_gain0 6.0) (> capital_gain0 28.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (<= martial_status0 1.0) (<= hours_per_week0 22.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (<= martial_status0 1.0) (> hours_per_week0 22.0) (<= race0 1.0) (<= native_country0 16.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (<= martial_status0 1.0) (> hours_per_week0 22.0) (<= race0 1.0) (> native_country0 16.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (<= martial_status0 1.0) (> hours_per_week0 22.0) (> race0 1.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (> martial_status0 1.0) (<= hours_per_week0 32.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (> martial_status0 1.0) (> hours_per_week0 32.0) (<= occupation0 0.0) (<= sex0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (> martial_status0 1.0) (> hours_per_week0 32.0) (<= occupation0 0.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (> martial_status0 1.0) (> hours_per_week0 32.0) (> occupation0 0.0) (<= fnlwgt0 7.0) (<= fnlwgt0 5.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (> martial_status0 1.0) (> hours_per_week0 32.0) (> occupation0 0.0) (<= fnlwgt0 7.0) (> fnlwgt0 5.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (<= occupation0 1.0) (> martial_status0 1.0) (> hours_per_week0 32.0) (> occupation0 0.0) (> fnlwgt0 7.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (<= hours_per_week0 27.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (> hours_per_week0 27.0) (<= relationship0 4.0) (<= age0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (> hours_per_week0 27.0) (<= relationship0 4.0) (> age0 3.0) (<= fnlwgt0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (> hours_per_week0 27.0) (<= relationship0 4.0) (> age0 3.0) (> fnlwgt0 3.0) (<= age0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (> hours_per_week0 27.0) (<= relationship0 4.0) (> age0 3.0) (> fnlwgt0 3.0) (> age0 4.0) (<= hours_per_week0 31.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (> hours_per_week0 27.0) (<= relationship0 4.0) (> age0 3.0) (> fnlwgt0 3.0) (> age0 4.0) (> hours_per_week0 31.0) (<= relationship0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (> hours_per_week0 27.0) (<= relationship0 4.0) (> age0 3.0) (> fnlwgt0 3.0) (> age0 4.0) (> hours_per_week0 31.0) (> relationship0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (<= martial_status0 1.0) (> hours_per_week0 27.0) (> relationship0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (<= capital_gain0 0.0) (> martial_status0 1.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (<= capital_loss0 7.0) (> occupation0 1.0) (> capital_gain0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (<= hours_per_week0 35.0) (> capital_loss0 7.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (<= age0 2.0) (<= occupation0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (<= age0 2.0) (> occupation0 2.0) (<= relationship0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (<= age0 2.0) (> occupation0 2.0) (> relationship0 3.0) (<= fnlwgt0 6.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (<= age0 2.0) (> occupation0 2.0) (> relationship0 3.0) (> fnlwgt0 6.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (> age0 2.0) (<= workclass0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (> age0 2.0) (> workclass0 0.0) (<= sex0 0.0) (<= hours_per_week0 45.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (> age0 2.0) (> workclass0 0.0) (<= sex0 0.0) (> hours_per_week0 45.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (<= workclass0 1.0) (> age0 2.0) (> workclass0 0.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (> workclass0 1.0) (<= hours_per_week0 42.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (<= fnlwgt0 28.0) (> workclass0 1.0) (> hours_per_week0 42.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (<= occupation0 3.0) (> fnlwgt0 28.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (<= fnlwgt0 16.0) (<= workclass0 0.0) (<= age0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (<= fnlwgt0 16.0) (<= workclass0 0.0) (> age0 2.0) (<= fnlwgt0 11.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (<= fnlwgt0 16.0) (<= workclass0 0.0) (> age0 2.0) (> fnlwgt0 11.0) (<= sex0 0.0) (<= education0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (<= fnlwgt0 16.0) (<= workclass0 0.0) (> age0 2.0) (> fnlwgt0 11.0) (<= sex0 0.0) (> education0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (<= fnlwgt0 16.0) (<= workclass0 0.0) (> age0 2.0) (> fnlwgt0 11.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (<= fnlwgt0 16.0) (> workclass0 0.0) (<= capital_loss0 9.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (<= fnlwgt0 16.0) (> workclass0 0.0) (> capital_loss0 9.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (<= relationship0 4.0) (> fnlwgt0 16.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (<= hours_per_week0 41.0) (> relationship0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (> hours_per_week0 41.0) (<= workclass0 1.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (> hours_per_week0 41.0) (> workclass0 1.0) (<= relationship0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (> hours_per_week0 41.0) (> workclass0 1.0) (> relationship0 4.0) (<= hours_per_week0 52.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (<= native_country0 4.0) (> occupation0 3.0) (> hours_per_week0 41.0) (> workclass0 1.0) (> relationship0 4.0) (> hours_per_week0 52.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (> native_country0 4.0) (<= hours_per_week0 65.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (<= workclass0 2.0) (> native_country0 4.0) (> hours_per_week0 65.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (> workclass0 2.0) (<= occupation0 0.0) (<= age0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (> workclass0 2.0) (<= occupation0 0.0) (> age0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (> workclass0 2.0) (> occupation0 0.0) (<= hours_per_week0 55.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (<= race0 0.0) (> workclass0 2.0) (> occupation0 0.0) (> hours_per_week0 55.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (> race0 0.0) (<= hours_per_week0 49.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (> race0 0.0) (> hours_per_week0 49.0) (<= fnlwgt0 10.0) (<= native_country0 4.0) (<= education0 1.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (> race0 0.0) (> hours_per_week0 49.0) (<= fnlwgt0 10.0) (<= native_country0 4.0) (> education0 1.0) (<= relationship0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (> race0 0.0) (> hours_per_week0 49.0) (<= fnlwgt0 10.0) (<= native_country0 4.0) (> education0 1.0) (> relationship0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (> race0 0.0) (> hours_per_week0 49.0) (<= fnlwgt0 10.0) (> native_country0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (<= martial_status0 1.0) (> race0 0.0) (> hours_per_week0 49.0) (> fnlwgt0 10.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (<= fnlwgt0 11.0) (<= hours_per_week0 36.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (<= fnlwgt0 11.0) (> hours_per_week0 36.0) (<= workclass0 0.0) (<= age0 1.0) (<= sex0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (<= fnlwgt0 11.0) (> hours_per_week0 36.0) (<= workclass0 0.0) (<= age0 1.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (<= fnlwgt0 11.0) (> hours_per_week0 36.0) (<= workclass0 0.0) (> age0 1.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (<= fnlwgt0 11.0) (> hours_per_week0 36.0) (> workclass0 0.0) (<= education0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (<= fnlwgt0 11.0) (> hours_per_week0 36.0) (> workclass0 0.0) (> education0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (<= workclass0 0.0) (<= occupation0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (<= workclass0 0.0) (> occupation0 0.0) (<= relationship0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (<= workclass0 0.0) (> occupation0 0.0) (> relationship0 2.0) (<= age0 2.0) (<= hours_per_week0 42.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (<= workclass0 0.0) (> occupation0 0.0) (> relationship0 2.0) (<= age0 2.0) (> hours_per_week0 42.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (<= workclass0 0.0) (> occupation0 0.0) (> relationship0 2.0) (> age0 2.0) (<= hours_per_week0 39.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (<= workclass0 0.0) (> occupation0 0.0) (> relationship0 2.0) (> age0 2.0) (> hours_per_week0 39.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (> workclass0 0.0) (<= hours_per_week0 47.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (<= native_country0 4.0) (> fnlwgt0 11.0) (> workclass0 0.0) (> hours_per_week0 47.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (> native_country0 4.0) (<= fnlwgt0 8.0) (<= capital_loss0 5.0) (<= education0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (> native_country0 4.0) (<= fnlwgt0 8.0) (<= capital_loss0 5.0) (> education0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (> native_country0 4.0) (<= fnlwgt0 8.0) (> capital_loss0 5.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (<= race0 1.0) (> native_country0 4.0) (> fnlwgt0 8.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (> race0 1.0) (<= age0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (<= relationship0 3.0) (> race0 1.0) (> age0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (<= native_country0 1.0) (<= hours_per_week0 45.0) (<= fnlwgt0 10.0) (<= occupation0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (<= native_country0 1.0) (<= hours_per_week0 45.0) (<= fnlwgt0 10.0) (> occupation0 0.0) (<= fnlwgt0 5.0) (<= age0 2.0) (<= fnlwgt0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (<= native_country0 1.0) (<= hours_per_week0 45.0) (<= fnlwgt0 10.0) (> occupation0 0.0) (<= fnlwgt0 5.0) (<= age0 2.0) (> fnlwgt0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (<= native_country0 1.0) (<= hours_per_week0 45.0) (<= fnlwgt0 10.0) (> occupation0 0.0) (<= fnlwgt0 5.0) (> age0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (<= native_country0 1.0) (<= hours_per_week0 45.0) (<= fnlwgt0 10.0) (> occupation0 0.0) (> fnlwgt0 5.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (<= native_country0 1.0) (<= hours_per_week0 45.0) (> fnlwgt0 10.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (<= native_country0 1.0) (> hours_per_week0 45.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (<= education0 4.0) (> native_country0 1.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (<= workclass0 1.0) (> relationship0 3.0) (> education0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (<= martial_status0 2.0) (> workclass0 1.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (> martial_status0 2.0) (<= capital_loss0 15.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (<= occupation0 1.0) (> martial_status0 2.0) (> capital_loss0 15.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (<= hours_per_week0 39.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (> hours_per_week0 39.0) (<= workclass0 2.0) (<= fnlwgt0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (> hours_per_week0 39.0) (<= workclass0 2.0) (> fnlwgt0 3.0) (<= sex0 0.0) (<= hours_per_week0 42.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (> hours_per_week0 39.0) (<= workclass0 2.0) (> fnlwgt0 3.0) (<= sex0 0.0) (> hours_per_week0 42.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (> hours_per_week0 39.0) (<= workclass0 2.0) (> fnlwgt0 3.0) (> sex0 0.0) (<= education0 1.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (> hours_per_week0 39.0) (<= workclass0 2.0) (> fnlwgt0 3.0) (> sex0 0.0) (> education0 1.0) (<= fnlwgt0 6.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (> hours_per_week0 39.0) (<= workclass0 2.0) (> fnlwgt0 3.0) (> sex0 0.0) (> education0 1.0) (> fnlwgt0 6.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (<= native_country0 4.0) (> hours_per_week0 39.0) (> workclass0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (<= race0 0.0) (> native_country0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (<= fnlwgt0 10.0) (> race0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (<= relationship0 2.0) (> fnlwgt0 10.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (<= hours_per_week0 42.0) (<= age0 6.0) (<= education0 1.0) (<= age0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (<= hours_per_week0 42.0) (<= age0 6.0) (<= education0 1.0) (> age0 3.0) (<= martial_status0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (<= hours_per_week0 42.0) (<= age0 6.0) (<= education0 1.0) (> age0 3.0) (> martial_status0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (<= hours_per_week0 42.0) (<= age0 6.0) (> education0 1.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (<= hours_per_week0 42.0) (> age0 6.0) (<= education0 8.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (<= hours_per_week0 42.0) (> age0 6.0) (> education0 8.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (> hours_per_week0 42.0) (<= age0 4.0) (<= education0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (> hours_per_week0 42.0) (<= age0 4.0) (> education0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (<= occupation0 2.0) (> relationship0 2.0) (> hours_per_week0 42.0) (> age0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (> occupation0 2.0) (<= hours_per_week0 44.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (> occupation0 2.0) (> hours_per_week0 44.0) (<= fnlwgt0 6.0) (<= occupation0 3.0) (<= sex0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (> occupation0 2.0) (> hours_per_week0 44.0) (<= fnlwgt0 6.0) (<= occupation0 3.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (> occupation0 2.0) (> hours_per_week0 44.0) (<= fnlwgt0 6.0) (> occupation0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (<= capital_loss0 3.0) (> occupation0 2.0) (> hours_per_week0 44.0) (> fnlwgt0 6.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (> capital_loss0 3.0) (<= native_country0 56.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (<= hours_per_week0 45.0) (> capital_loss0 3.0) (> native_country0 56.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (<= occupation0 3.0) (<= fnlwgt0 11.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (<= occupation0 3.0) (> fnlwgt0 11.0) (<= education0 4.0) (<= education0 2.0) (<= fnlwgt0 15.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (<= occupation0 3.0) (> fnlwgt0 11.0) (<= education0 4.0) (<= education0 2.0) (> fnlwgt0 15.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (<= occupation0 3.0) (> fnlwgt0 11.0) (<= education0 4.0) (> education0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (<= occupation0 3.0) (> fnlwgt0 11.0) (> education0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (> occupation0 3.0) (<= age0 3.0) (<= fnlwgt0 3.0) (<= sex0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (> occupation0 3.0) (<= age0 3.0) (<= fnlwgt0 3.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (> occupation0 3.0) (<= age0 3.0) (> fnlwgt0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (<= workclass0 0.0) (> occupation0 3.0) (> age0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (<= hours_per_week0 51.0) (> workclass0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (> hours_per_week0 51.0) (<= workclass0 1.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (> hours_per_week0 51.0) (> workclass0 1.0) (<= occupation0 3.0) (<= sex0 0.0) (<= hours_per_week0 67.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (> hours_per_week0 51.0) (> workclass0 1.0) (<= occupation0 3.0) (<= sex0 0.0) (> hours_per_week0 67.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (> hours_per_week0 51.0) (> workclass0 1.0) (<= occupation0 3.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (<= race0 3.0) (> hours_per_week0 51.0) (> workclass0 1.0) (> occupation0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (<= martial_status0 2.0) (> race0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (> martial_status0 2.0) (<= capital_loss0 9.0) (<= hours_per_week0 57.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (> martial_status0 2.0) (<= capital_loss0 9.0) (> hours_per_week0 57.0) (<= fnlwgt0 11.0) (<= fnlwgt0 4.0) (<= martial_status0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (> martial_status0 2.0) (<= capital_loss0 9.0) (> hours_per_week0 57.0) (<= fnlwgt0 11.0) (<= fnlwgt0 4.0) (> martial_status0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (> martial_status0 2.0) (<= capital_loss0 9.0) (> hours_per_week0 57.0) (<= fnlwgt0 11.0) (> fnlwgt0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (> martial_status0 2.0) (<= capital_loss0 9.0) (> hours_per_week0 57.0) (> fnlwgt0 11.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (<= native_country0 3.0) (> martial_status0 2.0) (> capital_loss0 9.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (<= occupation0 4.0) (> hours_per_week0 35.0) (> martial_status0 1.0) (> occupation0 1.0) (> hours_per_week0 45.0) (> native_country0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (<= fnlwgt0 5.0) (<= relationship0 3.0) (<= hours_per_week0 39.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (<= fnlwgt0 5.0) (<= relationship0 3.0) (> hours_per_week0 39.0) (<= workclass0 1.0) (<= education0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (<= fnlwgt0 5.0) (<= relationship0 3.0) (> hours_per_week0 39.0) (<= workclass0 1.0) (> education0 4.0) (<= sex0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (<= fnlwgt0 5.0) (<= relationship0 3.0) (> hours_per_week0 39.0) (<= workclass0 1.0) (> education0 4.0) (> sex0 0.0) (<= education0 11.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (<= fnlwgt0 5.0) (<= relationship0 3.0) (> hours_per_week0 39.0) (<= workclass0 1.0) (> education0 4.0) (> sex0 0.0) (> education0 11.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (<= fnlwgt0 5.0) (<= relationship0 3.0) (> hours_per_week0 39.0) (> workclass0 1.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (<= fnlwgt0 5.0) (> relationship0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (> fnlwgt0 5.0) (<= education0 0.0) (<= fnlwgt0 8.0) (<= workclass0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (> fnlwgt0 5.0) (<= education0 0.0) (<= fnlwgt0 8.0) (> workclass0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (> fnlwgt0 5.0) (<= education0 0.0) (> fnlwgt0 8.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (<= hours_per_week0 41.0) (> fnlwgt0 5.0) (> education0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (> hours_per_week0 41.0) (<= workclass0 2.0) (<= age0 3.0) (<= education0 2.0) (<= fnlwgt0 19.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (> hours_per_week0 41.0) (<= workclass0 2.0) (<= age0 3.0) (<= education0 2.0) (> fnlwgt0 19.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (> hours_per_week0 41.0) (<= workclass0 2.0) (<= age0 3.0) (> education0 2.0) (<= native_country0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (> hours_per_week0 41.0) (<= workclass0 2.0) (<= age0 3.0) (> education0 2.0) (> native_country0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (> hours_per_week0 41.0) (<= workclass0 2.0) (> age0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (<= occupation0 6.0) (> hours_per_week0 41.0) (> workclass0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (> occupation0 6.0) (<= hours_per_week0 49.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (> occupation0 6.0) (> hours_per_week0 49.0) (<= occupation0 7.0) (<= race0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (> occupation0 6.0) (> hours_per_week0 49.0) (<= occupation0 7.0) (> race0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (<= martial_status0 1.0) (> occupation0 6.0) (> hours_per_week0 49.0) (> occupation0 7.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (> martial_status0 1.0) (<= capital_gain0 0.0) (<= education0 12.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (> martial_status0 1.0) (<= capital_gain0 0.0) (> education0 12.0) (<= fnlwgt0 1.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (> martial_status0 1.0) (<= capital_gain0 0.0) (> education0 12.0) (> fnlwgt0 1.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (> martial_status0 1.0) (> capital_gain0 0.0) (<= race0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (<= hours_per_week0 54.0) (> martial_status0 1.0) (> capital_gain0 0.0) (> race0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (<= martial_status0 1.0) (<= hours_per_week0 55.0) (<= fnlwgt0 7.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (<= martial_status0 1.0) (<= hours_per_week0 55.0) (> fnlwgt0 7.0) (<= occupation0 9.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (<= martial_status0 1.0) (<= hours_per_week0 55.0) (> fnlwgt0 7.0) (> occupation0 9.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (<= martial_status0 1.0) (> hours_per_week0 55.0) (<= fnlwgt0 8.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (<= martial_status0 1.0) (> hours_per_week0 55.0) (> fnlwgt0 8.0) (<= hours_per_week0 67.0) (<= occupation0 9.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (<= martial_status0 1.0) (> hours_per_week0 55.0) (> fnlwgt0 8.0) (<= hours_per_week0 67.0) (> occupation0 9.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (<= martial_status0 1.0) (> hours_per_week0 55.0) (> fnlwgt0 8.0) (> hours_per_week0 67.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (<= occupation0 7.0) (<= hours_per_week0 58.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (<= occupation0 7.0) (> hours_per_week0 58.0) (<= native_country0 5.0) (<= workclass0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (<= occupation0 7.0) (> hours_per_week0 58.0) (<= native_country0 5.0) (> workclass0 0.0) (<= fnlwgt0 6.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (<= occupation0 7.0) (> hours_per_week0 58.0) (<= native_country0 5.0) (> workclass0 0.0) (> fnlwgt0 6.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (<= occupation0 7.0) (> hours_per_week0 58.0) (> native_country0 5.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (> occupation0 7.0) (<= hours_per_week0 82.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (> occupation0 7.0) (> hours_per_week0 82.0) (<= age0 4.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (<= race0 0.0) (> martial_status0 1.0) (> occupation0 7.0) (> hours_per_week0 82.0) (> age0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (<= workclass0 2.0) (> race0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (<= capital_loss0 5.0) (> hours_per_week0 54.0) (> workclass0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (> capital_loss0 5.0) (<= workclass0 23.0) (<= native_country0 23.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (> capital_loss0 5.0) (<= workclass0 23.0) (> native_country0 23.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (<= capital_gain0 1.0) (> occupation0 4.0) (> capital_loss0 5.0) (> workclass0 23.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (<= native_country0 6.0) (<= occupation0 10.0) (<= capital_gain0 2.0) (<= race0 2.0) (<= martial_status0 3.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (<= native_country0 6.0) (<= occupation0 10.0) (<= capital_gain0 2.0) (<= race0 2.0) (> martial_status0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (<= native_country0 6.0) (<= occupation0 10.0) (<= capital_gain0 2.0) (> race0 2.0) (<= occupation0 6.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (<= native_country0 6.0) (<= occupation0 10.0) (<= capital_gain0 2.0) (> race0 2.0) (> occupation0 6.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (<= native_country0 6.0) (<= occupation0 10.0) (> capital_gain0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (<= native_country0 6.0) (> occupation0 10.0) (<= capital_loss0 7.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (<= native_country0 6.0) (> occupation0 10.0) (> capital_loss0 7.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (<= capital_gain0 9.0) (<= capital_loss0 21.0) (<= age0 3.0) (<= fnlwgt0 14.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (<= capital_gain0 9.0) (<= capital_loss0 21.0) (<= age0 3.0) (> fnlwgt0 14.0) (<= native_country0 48.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (<= capital_gain0 9.0) (<= capital_loss0 21.0) (<= age0 3.0) (> fnlwgt0 14.0) (> native_country0 48.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (<= capital_gain0 9.0) (<= capital_loss0 21.0) (> age0 3.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (<= capital_gain0 9.0) (> capital_loss0 21.0) (<= education0 4.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (<= capital_gain0 9.0) (> capital_loss0 21.0) (> education0 4.0) (<= fnlwgt0 60.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (<= capital_gain0 9.0) (> capital_loss0 21.0) (> education0 4.0) (> fnlwgt0 60.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (> capital_gain0 9.0) (<= workclass0 31.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (> capital_gain0 9.0) (> workclass0 31.0) (<= capital_loss0 12.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (<= workclass0 36.0) (> native_country0 6.0) (> capital_gain0 9.0) (> workclass0 31.0) (> capital_loss0 12.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (<= capital_gain0 14.0) (<= occupation0 38.0) (<= hours_per_week0 62.0) (<= occupation0 36.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (<= capital_gain0 14.0) (<= occupation0 38.0) (<= hours_per_week0 62.0) (> occupation0 36.0) (<= education0 12.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (<= capital_gain0 14.0) (<= occupation0 38.0) (<= hours_per_week0 62.0) (> occupation0 36.0) (> education0 12.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (<= capital_gain0 14.0) (<= occupation0 38.0) (> hours_per_week0 62.0) (<= hours_per_week0 76.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (<= capital_gain0 14.0) (<= occupation0 38.0) (> hours_per_week0 62.0) (> hours_per_week0 76.0) (<= age0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (<= capital_gain0 14.0) (<= occupation0 38.0) (> hours_per_week0 62.0) (> hours_per_week0 76.0) (> age0 2.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (<= capital_gain0 14.0) (> occupation0 38.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (<= workclass0 57.0) (<= occupation0 70.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (<= workclass0 57.0) (> occupation0 70.0) (<= native_country0 51.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (<= workclass0 57.0) (> occupation0 70.0) (> native_country0 51.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (> workclass0 57.0) (<= occupation0 49.0) (<= education0 10.0) (<= education0 2.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (> workclass0 57.0) (<= occupation0 49.0) (<= education0 10.0) (> education0 2.0) (<= occupation0 16.0) (<= sex0 0.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (> workclass0 57.0) (<= occupation0 49.0) (<= education0 10.0) (> education0 2.0) (<= occupation0 16.0) (> sex0 0.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (> workclass0 57.0) (<= occupation0 49.0) (<= education0 10.0) (> education0 2.0) (> occupation0 16.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (> workclass0 57.0) (<= occupation0 49.0) (> education0 10.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (<= capital_loss0 34.0) (> workclass0 57.0) (> occupation0 49.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (<= capital_gain0 24.0) (> workclass0 36.0) (> capital_gain0 14.0) (> capital_loss0 34.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (> capital_gain0 24.0) (<= capital_gain0 32.0) (<= native_country0 63.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (> capital_gain0 24.0) (<= capital_gain0 32.0) (> native_country0 63.0) (<= workclass0 62.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (> capital_gain0 24.0) (<= capital_gain0 32.0) (> native_country0 63.0) (> workclass0 62.0) (<= occupation0 55.0) ) (= Class0 1)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (> capital_gain0 24.0) (<= capital_gain0 32.0) (> native_country0 63.0) (> workclass0 62.0) (> occupation0 55.0) ) (= Class0 0)))
(assert (=> (and (> martial_status0 0.0) (> capital_gain0 1.0) (> capital_gain0 24.0) (> capital_gain0 32.0) ) (= Class0 1)))

;-----------1-----------number instance--------------
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (<= hours_per_week1 11.0) (<= capital_gain1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (<= hours_per_week1 11.0) (> capital_gain1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (<= hours_per_week1 21.0) (<= occupation1 2.0) (<= hours_per_week1 17.0) (<= workclass1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (<= hours_per_week1 21.0) (<= occupation1 2.0) (<= hours_per_week1 17.0) (> workclass1 0.0) (<= age1 5.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (<= hours_per_week1 21.0) (<= occupation1 2.0) (<= hours_per_week1 17.0) (> workclass1 0.0) (> age1 5.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (<= hours_per_week1 21.0) (<= occupation1 2.0) (> hours_per_week1 17.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (<= hours_per_week1 21.0) (> occupation1 2.0) (<= fnlwgt1 7.0) (<= workclass1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (<= hours_per_week1 21.0) (> occupation1 2.0) (<= fnlwgt1 7.0) (> workclass1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (<= hours_per_week1 21.0) (> occupation1 2.0) (> fnlwgt1 7.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (<= race1 0.0) (> hours_per_week1 21.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (<= occupation1 3.0) (> hours_per_week1 11.0) (> race1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (<= hours_per_week1 19.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (> hours_per_week1 19.0) (<= occupation1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (> hours_per_week1 19.0) (> occupation1 4.0) (<= hours_per_week1 24.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (> hours_per_week1 19.0) (> occupation1 4.0) (> hours_per_week1 24.0) (<= occupation1 5.0) (<= education1 2.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (> hours_per_week1 19.0) (> occupation1 4.0) (> hours_per_week1 24.0) (<= occupation1 5.0) (> education1 2.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (> hours_per_week1 19.0) (> occupation1 4.0) (> hours_per_week1 24.0) (> occupation1 5.0) (<= age1 5.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (> hours_per_week1 19.0) (> occupation1 4.0) (> hours_per_week1 24.0) (> occupation1 5.0) (> age1 5.0) (<= hours_per_week1 28.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (<= workclass1 0.0) (> hours_per_week1 19.0) (> occupation1 4.0) (> hours_per_week1 24.0) (> occupation1 5.0) (> age1 5.0) (> hours_per_week1 28.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (<= capital_loss1 9.0) (> workclass1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (<= capital_gain1 0.0) (> capital_loss1 9.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (<= hours_per_week1 31.0) (> occupation1 3.0) (> capital_gain1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (<= occupation1 4.0) (<= native_country1 5.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (<= occupation1 4.0) (> native_country1 5.0) (<= native_country1 7.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (<= occupation1 4.0) (> native_country1 5.0) (> native_country1 7.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (<= hours_per_week1 39.0) (<= workclass1 0.0) (<= fnlwgt1 27.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (<= hours_per_week1 39.0) (<= workclass1 0.0) (> fnlwgt1 27.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (<= hours_per_week1 39.0) (> workclass1 0.0) (<= fnlwgt1 6.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (<= hours_per_week1 39.0) (> workclass1 0.0) (> fnlwgt1 6.0) (<= education1 3.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (<= hours_per_week1 39.0) (> workclass1 0.0) (> fnlwgt1 6.0) (> education1 3.0) (<= fnlwgt1 8.0) (<= hours_per_week1 36.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (<= hours_per_week1 39.0) (> workclass1 0.0) (> fnlwgt1 6.0) (> education1 3.0) (<= fnlwgt1 8.0) (> hours_per_week1 36.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (<= hours_per_week1 39.0) (> workclass1 0.0) (> fnlwgt1 6.0) (> education1 3.0) (> fnlwgt1 8.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (<= native_country1 3.0) (<= workclass1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (<= native_country1 3.0) (> workclass1 0.0) (<= occupation1 5.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (<= native_country1 3.0) (> workclass1 0.0) (> occupation1 5.0) (<= hours_per_week1 42.0) (<= workclass1 1.0) (<= fnlwgt1 3.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (<= native_country1 3.0) (> workclass1 0.0) (> occupation1 5.0) (<= hours_per_week1 42.0) (<= workclass1 1.0) (> fnlwgt1 3.0) (<= occupation1 6.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (<= native_country1 3.0) (> workclass1 0.0) (> occupation1 5.0) (<= hours_per_week1 42.0) (<= workclass1 1.0) (> fnlwgt1 3.0) (> occupation1 6.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (<= native_country1 3.0) (> workclass1 0.0) (> occupation1 5.0) (<= hours_per_week1 42.0) (> workclass1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (<= native_country1 3.0) (> workclass1 0.0) (> occupation1 5.0) (> hours_per_week1 42.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (> native_country1 3.0) (<= hours_per_week1 43.0) (<= occupation1 6.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (> native_country1 3.0) (<= hours_per_week1 43.0) (> occupation1 6.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (<= race1 0.0) (> occupation1 4.0) (> hours_per_week1 39.0) (> native_country1 3.0) (> hours_per_week1 43.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (<= occupation1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (<= race1 3.0) (<= native_country1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (<= race1 3.0) (> native_country1 4.0) (<= hours_per_week1 45.0) (<= capital_gain1 2.0) (<= fnlwgt1 14.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (<= race1 3.0) (> native_country1 4.0) (<= hours_per_week1 45.0) (<= capital_gain1 2.0) (> fnlwgt1 14.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (<= race1 3.0) (> native_country1 4.0) (<= hours_per_week1 45.0) (> capital_gain1 2.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (<= race1 3.0) (> native_country1 4.0) (> hours_per_week1 45.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (<= occupation1 2.0) (<= age1 3.0) (<= fnlwgt1 8.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (<= occupation1 2.0) (<= age1 3.0) (> fnlwgt1 8.0) (<= education1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (<= occupation1 2.0) (<= age1 3.0) (> fnlwgt1 8.0) (> education1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (<= occupation1 2.0) (> age1 3.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (> occupation1 2.0) (<= age1 2.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (> occupation1 2.0) (> age1 2.0) (<= education1 2.0) (<= hours_per_week1 45.0) (<= age1 5.0) (<= fnlwgt1 21.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (> occupation1 2.0) (> age1 2.0) (<= education1 2.0) (<= hours_per_week1 45.0) (<= age1 5.0) (> fnlwgt1 21.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (> occupation1 2.0) (> age1 2.0) (<= education1 2.0) (<= hours_per_week1 45.0) (> age1 5.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (> occupation1 2.0) (> age1 2.0) (<= education1 2.0) (> hours_per_week1 45.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (> occupation1 2.0) (> age1 2.0) (> education1 2.0) (<= sex1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (<= hours_per_week1 55.0) (> occupation1 2.0) (> age1 2.0) (> education1 2.0) (> sex1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (<= occupation1 5.0) (> occupation1 1.0) (> race1 3.0) (> hours_per_week1 55.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (> occupation1 5.0) (<= capital_gain1 1.0) (<= hours_per_week1 55.0) (<= capital_loss1 9.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (> occupation1 5.0) (<= capital_gain1 1.0) (<= hours_per_week1 55.0) (> capital_loss1 9.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (> occupation1 5.0) (<= capital_gain1 1.0) (> hours_per_week1 55.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (<= occupation1 7.0) (> hours_per_week1 31.0) (> race1 0.0) (> occupation1 5.0) (> capital_gain1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (<= hours_per_week1 35.0) (<= capital_gain1 1.0) (<= capital_loss1 8.0) (<= hours_per_week1 32.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (<= hours_per_week1 35.0) (<= capital_gain1 1.0) (<= capital_loss1 8.0) (> hours_per_week1 32.0) (<= fnlwgt1 6.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (<= hours_per_week1 35.0) (<= capital_gain1 1.0) (<= capital_loss1 8.0) (> hours_per_week1 32.0) (> fnlwgt1 6.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (<= hours_per_week1 35.0) (<= capital_gain1 1.0) (> capital_loss1 8.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (<= hours_per_week1 35.0) (> capital_gain1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (<= fnlwgt1 10.0) (<= native_country1 1.0) (<= age1 2.0) (<= fnlwgt1 5.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (<= fnlwgt1 10.0) (<= native_country1 1.0) (<= age1 2.0) (> fnlwgt1 5.0) (<= sex1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (<= fnlwgt1 10.0) (<= native_country1 1.0) (<= age1 2.0) (> fnlwgt1 5.0) (> sex1 0.0) (<= capital_gain1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (<= fnlwgt1 10.0) (<= native_country1 1.0) (<= age1 2.0) (> fnlwgt1 5.0) (> sex1 0.0) (> capital_gain1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (<= fnlwgt1 10.0) (<= native_country1 1.0) (> age1 2.0) (<= workclass1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (<= fnlwgt1 10.0) (<= native_country1 1.0) (> age1 2.0) (> workclass1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (<= fnlwgt1 10.0) (> native_country1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (<= hours_per_week1 41.0) (<= sex1 0.0) (<= fnlwgt1 15.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (<= hours_per_week1 41.0) (<= sex1 0.0) (> fnlwgt1 15.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (<= hours_per_week1 41.0) (> sex1 0.0) (<= age1 3.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (<= hours_per_week1 41.0) (> sex1 0.0) (> age1 3.0) (<= fnlwgt1 14.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (<= hours_per_week1 41.0) (> sex1 0.0) (> age1 3.0) (> fnlwgt1 14.0) (<= age1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (<= hours_per_week1 41.0) (> sex1 0.0) (> age1 3.0) (> fnlwgt1 14.0) (> age1 4.0) (<= fnlwgt1 24.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (<= hours_per_week1 41.0) (> sex1 0.0) (> age1 3.0) (> fnlwgt1 14.0) (> age1 4.0) (> fnlwgt1 24.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (<= race1 0.0) (> hours_per_week1 35.0) (> fnlwgt1 10.0) (> hours_per_week1 41.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (> race1 0.0) (<= capital_gain1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (<= occupation1 8.0) (> race1 0.0) (> capital_gain1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (<= hours_per_week1 44.0) (<= fnlwgt1 1.0) (<= workclass1 0.0) (<= occupation1 9.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (<= hours_per_week1 44.0) (<= fnlwgt1 1.0) (<= workclass1 0.0) (> occupation1 9.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (<= hours_per_week1 44.0) (<= fnlwgt1 1.0) (> workclass1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (<= hours_per_week1 44.0) (> fnlwgt1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (<= workclass1 0.0) (<= fnlwgt1 10.0) (<= education1 2.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (<= workclass1 0.0) (<= fnlwgt1 10.0) (> education1 2.0) (<= race1 0.0) (<= fnlwgt1 7.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (<= workclass1 0.0) (<= fnlwgt1 10.0) (> education1 2.0) (<= race1 0.0) (> fnlwgt1 7.0) (<= age1 3.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (<= workclass1 0.0) (<= fnlwgt1 10.0) (> education1 2.0) (<= race1 0.0) (> fnlwgt1 7.0) (> age1 3.0) (<= native_country1 2.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (<= workclass1 0.0) (<= fnlwgt1 10.0) (> education1 2.0) (<= race1 0.0) (> fnlwgt1 7.0) (> age1 3.0) (> native_country1 2.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (<= workclass1 0.0) (<= fnlwgt1 10.0) (> education1 2.0) (> race1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (<= workclass1 0.0) (> fnlwgt1 10.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (<= capital_loss1 9.0) (> hours_per_week1 44.0) (> workclass1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (<= capital_gain1 1.0) (> capital_loss1 9.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (<= hours_per_week1 49.0) (> occupation1 8.0) (> capital_gain1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (<= fnlwgt1 17.0) (<= workclass1 0.0) (<= education1 12.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (<= fnlwgt1 17.0) (<= workclass1 0.0) (> education1 12.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (<= fnlwgt1 17.0) (> workclass1 0.0) (<= fnlwgt1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (<= fnlwgt1 17.0) (> workclass1 0.0) (> fnlwgt1 4.0) (<= age1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (<= fnlwgt1 17.0) (> workclass1 0.0) (> fnlwgt1 4.0) (> age1 4.0) (<= fnlwgt1 6.0) (<= workclass1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (<= fnlwgt1 17.0) (> workclass1 0.0) (> fnlwgt1 4.0) (> age1 4.0) (<= fnlwgt1 6.0) (> workclass1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (<= fnlwgt1 17.0) (> workclass1 0.0) (> fnlwgt1 4.0) (> age1 4.0) (> fnlwgt1 6.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (<= hours_per_week1 50.0) (> fnlwgt1 17.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (<= native_country1 9.0) (> occupation1 7.0) (> hours_per_week1 49.0) (> hours_per_week1 50.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (<= native_country1 14.0) (<= occupation1 2.0) (<= workclass1 0.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (<= native_country1 14.0) (<= occupation1 2.0) (> workclass1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (<= native_country1 14.0) (> occupation1 2.0) (<= hours_per_week1 49.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (<= native_country1 14.0) (> occupation1 2.0) (> hours_per_week1 49.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (> native_country1 14.0) (<= occupation1 1.0) (<= capital_loss1 7.0) (<= native_country1 18.0) (<= age1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (> native_country1 14.0) (<= occupation1 1.0) (<= capital_loss1 7.0) (<= native_country1 18.0) (> age1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (> native_country1 14.0) (<= occupation1 1.0) (<= capital_loss1 7.0) (> native_country1 18.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (> native_country1 14.0) (<= occupation1 1.0) (> capital_loss1 7.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (<= capital_loss1 15.0) (> native_country1 14.0) (> occupation1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (<= hours_per_week1 52.0) (> capital_loss1 15.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (> hours_per_week1 52.0) (<= native_country1 29.0) (<= fnlwgt1 26.0) (<= race1 2.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (> hours_per_week1 52.0) (<= native_country1 29.0) (<= fnlwgt1 26.0) (> race1 2.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (> hours_per_week1 52.0) (<= native_country1 29.0) (> fnlwgt1 26.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (<= capital_gain1 1.0) (> hours_per_week1 52.0) (> native_country1 29.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (> capital_gain1 1.0) (<= native_country1 78.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (> capital_gain1 1.0) (> native_country1 78.0) (<= education1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (> capital_gain1 1.0) (> native_country1 78.0) (> education1 1.0) (<= fnlwgt1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (<= workclass1 2.0) (> native_country1 9.0) (> capital_gain1 1.0) (> native_country1 78.0) (> education1 1.0) (> fnlwgt1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (<= occupation1 1.0) (<= race1 0.0) (<= workclass1 4.0) (<= hours_per_week1 27.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (<= occupation1 1.0) (<= race1 0.0) (<= workclass1 4.0) (> hours_per_week1 27.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (<= occupation1 1.0) (<= race1 0.0) (> workclass1 4.0) (<= fnlwgt1 5.0) (<= education1 5.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (<= occupation1 1.0) (<= race1 0.0) (> workclass1 4.0) (<= fnlwgt1 5.0) (> education1 5.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (<= occupation1 1.0) (<= race1 0.0) (> workclass1 4.0) (> fnlwgt1 5.0) (<= age1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (<= occupation1 1.0) (<= race1 0.0) (> workclass1 4.0) (> fnlwgt1 5.0) (> age1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (<= occupation1 1.0) (> race1 0.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (<= workclass1 4.0) (<= native_country1 5.0) (<= hours_per_week1 35.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (<= workclass1 4.0) (<= native_country1 5.0) (> hours_per_week1 35.0) (<= age1 4.0) (<= fnlwgt1 3.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (<= workclass1 4.0) (<= native_country1 5.0) (> hours_per_week1 35.0) (<= age1 4.0) (> fnlwgt1 3.0) (<= education1 2.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (<= workclass1 4.0) (<= native_country1 5.0) (> hours_per_week1 35.0) (<= age1 4.0) (> fnlwgt1 3.0) (> education1 2.0) (<= workclass1 3.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (<= workclass1 4.0) (<= native_country1 5.0) (> hours_per_week1 35.0) (<= age1 4.0) (> fnlwgt1 3.0) (> education1 2.0) (> workclass1 3.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (<= workclass1 4.0) (<= native_country1 5.0) (> hours_per_week1 35.0) (> age1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (<= workclass1 4.0) (> native_country1 5.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (<= occupation1 3.0) (> workclass1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (<= hours_per_week1 47.0) (<= workclass1 3.0) (<= occupation1 4.0) (<= race1 1.0) (<= education1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (<= hours_per_week1 47.0) (<= workclass1 3.0) (<= occupation1 4.0) (<= race1 1.0) (> education1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (<= hours_per_week1 47.0) (<= workclass1 3.0) (<= occupation1 4.0) (> race1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (<= hours_per_week1 47.0) (<= workclass1 3.0) (> occupation1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (<= hours_per_week1 47.0) (> workclass1 3.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (<= workclass1 4.0) (<= fnlwgt1 16.0) (<= workclass1 3.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (<= workclass1 4.0) (<= fnlwgt1 16.0) (> workclass1 3.0) (<= age1 4.0) (<= occupation1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (<= workclass1 4.0) (<= fnlwgt1 16.0) (> workclass1 3.0) (<= age1 4.0) (> occupation1 4.0) (<= relationship1 1.0) (<= fnlwgt1 12.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (<= workclass1 4.0) (<= fnlwgt1 16.0) (> workclass1 3.0) (<= age1 4.0) (> occupation1 4.0) (<= relationship1 1.0) (> fnlwgt1 12.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (<= workclass1 4.0) (<= fnlwgt1 16.0) (> workclass1 3.0) (<= age1 4.0) (> occupation1 4.0) (> relationship1 1.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (<= workclass1 4.0) (<= fnlwgt1 16.0) (> workclass1 3.0) (> age1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (<= workclass1 4.0) (> fnlwgt1 16.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (<= capital_gain1 1.0) (> occupation1 3.0) (> hours_per_week1 47.0) (> workclass1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (<= hours_per_week1 51.0) (> capital_gain1 1.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (> hours_per_week1 51.0) (<= native_country1 2.0) (<= education1 11.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (> hours_per_week1 51.0) (<= native_country1 2.0) (> education1 11.0) (<= age1 4.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (> hours_per_week1 51.0) (<= native_country1 2.0) (> education1 11.0) (> age1 4.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (<= capital_loss1 6.0) (> occupation1 1.0) (> hours_per_week1 51.0) (> native_country1 2.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (<= occupation1 7.0) (> capital_loss1 6.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (<= capital_loss1 13.0) (<= capital_gain1 1.0) (<= hours_per_week1 58.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (<= capital_loss1 13.0) (<= capital_gain1 1.0) (> hours_per_week1 58.0) (<= workclass1 3.0) (<= race1 2.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (<= capital_loss1 13.0) (<= capital_gain1 1.0) (> hours_per_week1 58.0) (<= workclass1 3.0) (> race1 2.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (<= capital_loss1 13.0) (<= capital_gain1 1.0) (> hours_per_week1 58.0) (> workclass1 3.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (<= capital_loss1 13.0) (> capital_gain1 1.0) (<= workclass1 5.0) (<= age1 6.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (<= capital_loss1 13.0) (> capital_gain1 1.0) (<= workclass1 5.0) (> age1 6.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (<= capital_loss1 13.0) (> capital_gain1 1.0) (> workclass1 5.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (> capital_loss1 13.0) (<= workclass1 24.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (<= capital_gain1 6.0) (> occupation1 7.0) (> capital_loss1 13.0) (> workclass1 24.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (> capital_gain1 6.0) (<= capital_gain1 28.0) (<= workclass1 37.0) (<= native_country1 93.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (> capital_gain1 6.0) (<= capital_gain1 28.0) (<= workclass1 37.0) (> native_country1 93.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (> capital_gain1 6.0) (<= capital_gain1 28.0) (> workclass1 37.0) (<= occupation1 23.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (> capital_gain1 6.0) (<= capital_gain1 28.0) (> workclass1 37.0) (> occupation1 23.0) (<= capital_loss1 39.0) ) (= Class1 0)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (> capital_gain1 6.0) (<= capital_gain1 28.0) (> workclass1 37.0) (> occupation1 23.0) (> capital_loss1 39.0) ) (= Class1 1)))
(assert (=> (and (<= martial_status1 0.0) (> workclass1 2.0) (> capital_gain1 6.0) (> capital_gain1 28.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (<= martial_status1 1.0) (<= hours_per_week1 22.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (<= martial_status1 1.0) (> hours_per_week1 22.0) (<= race1 1.0) (<= native_country1 16.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (<= martial_status1 1.0) (> hours_per_week1 22.0) (<= race1 1.0) (> native_country1 16.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (<= martial_status1 1.0) (> hours_per_week1 22.0) (> race1 1.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (> martial_status1 1.0) (<= hours_per_week1 32.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (> martial_status1 1.0) (> hours_per_week1 32.0) (<= occupation1 0.0) (<= sex1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (> martial_status1 1.0) (> hours_per_week1 32.0) (<= occupation1 0.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (> martial_status1 1.0) (> hours_per_week1 32.0) (> occupation1 0.0) (<= fnlwgt1 7.0) (<= fnlwgt1 5.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (> martial_status1 1.0) (> hours_per_week1 32.0) (> occupation1 0.0) (<= fnlwgt1 7.0) (> fnlwgt1 5.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (<= occupation1 1.0) (> martial_status1 1.0) (> hours_per_week1 32.0) (> occupation1 0.0) (> fnlwgt1 7.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (<= hours_per_week1 27.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (> hours_per_week1 27.0) (<= relationship1 4.0) (<= age1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (> hours_per_week1 27.0) (<= relationship1 4.0) (> age1 3.0) (<= fnlwgt1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (> hours_per_week1 27.0) (<= relationship1 4.0) (> age1 3.0) (> fnlwgt1 3.0) (<= age1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (> hours_per_week1 27.0) (<= relationship1 4.0) (> age1 3.0) (> fnlwgt1 3.0) (> age1 4.0) (<= hours_per_week1 31.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (> hours_per_week1 27.0) (<= relationship1 4.0) (> age1 3.0) (> fnlwgt1 3.0) (> age1 4.0) (> hours_per_week1 31.0) (<= relationship1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (> hours_per_week1 27.0) (<= relationship1 4.0) (> age1 3.0) (> fnlwgt1 3.0) (> age1 4.0) (> hours_per_week1 31.0) (> relationship1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (<= martial_status1 1.0) (> hours_per_week1 27.0) (> relationship1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (<= capital_gain1 0.0) (> martial_status1 1.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (<= capital_loss1 7.0) (> occupation1 1.0) (> capital_gain1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (<= hours_per_week1 35.0) (> capital_loss1 7.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (<= age1 2.0) (<= occupation1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (<= age1 2.0) (> occupation1 2.0) (<= relationship1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (<= age1 2.0) (> occupation1 2.0) (> relationship1 3.0) (<= fnlwgt1 6.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (<= age1 2.0) (> occupation1 2.0) (> relationship1 3.0) (> fnlwgt1 6.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (> age1 2.0) (<= workclass1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (> age1 2.0) (> workclass1 0.0) (<= sex1 0.0) (<= hours_per_week1 45.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (> age1 2.0) (> workclass1 0.0) (<= sex1 0.0) (> hours_per_week1 45.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (<= workclass1 1.0) (> age1 2.0) (> workclass1 0.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (> workclass1 1.0) (<= hours_per_week1 42.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (<= fnlwgt1 28.0) (> workclass1 1.0) (> hours_per_week1 42.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (<= occupation1 3.0) (> fnlwgt1 28.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (<= fnlwgt1 16.0) (<= workclass1 0.0) (<= age1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (<= fnlwgt1 16.0) (<= workclass1 0.0) (> age1 2.0) (<= fnlwgt1 11.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (<= fnlwgt1 16.0) (<= workclass1 0.0) (> age1 2.0) (> fnlwgt1 11.0) (<= sex1 0.0) (<= education1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (<= fnlwgt1 16.0) (<= workclass1 0.0) (> age1 2.0) (> fnlwgt1 11.0) (<= sex1 0.0) (> education1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (<= fnlwgt1 16.0) (<= workclass1 0.0) (> age1 2.0) (> fnlwgt1 11.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (<= fnlwgt1 16.0) (> workclass1 0.0) (<= capital_loss1 9.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (<= fnlwgt1 16.0) (> workclass1 0.0) (> capital_loss1 9.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (<= relationship1 4.0) (> fnlwgt1 16.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (<= hours_per_week1 41.0) (> relationship1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (> hours_per_week1 41.0) (<= workclass1 1.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (> hours_per_week1 41.0) (> workclass1 1.0) (<= relationship1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (> hours_per_week1 41.0) (> workclass1 1.0) (> relationship1 4.0) (<= hours_per_week1 52.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (<= native_country1 4.0) (> occupation1 3.0) (> hours_per_week1 41.0) (> workclass1 1.0) (> relationship1 4.0) (> hours_per_week1 52.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (> native_country1 4.0) (<= hours_per_week1 65.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (<= workclass1 2.0) (> native_country1 4.0) (> hours_per_week1 65.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (> workclass1 2.0) (<= occupation1 0.0) (<= age1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (> workclass1 2.0) (<= occupation1 0.0) (> age1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (> workclass1 2.0) (> occupation1 0.0) (<= hours_per_week1 55.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (<= race1 0.0) (> workclass1 2.0) (> occupation1 0.0) (> hours_per_week1 55.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (> race1 0.0) (<= hours_per_week1 49.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (> race1 0.0) (> hours_per_week1 49.0) (<= fnlwgt1 10.0) (<= native_country1 4.0) (<= education1 1.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (> race1 0.0) (> hours_per_week1 49.0) (<= fnlwgt1 10.0) (<= native_country1 4.0) (> education1 1.0) (<= relationship1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (> race1 0.0) (> hours_per_week1 49.0) (<= fnlwgt1 10.0) (<= native_country1 4.0) (> education1 1.0) (> relationship1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (> race1 0.0) (> hours_per_week1 49.0) (<= fnlwgt1 10.0) (> native_country1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (<= martial_status1 1.0) (> race1 0.0) (> hours_per_week1 49.0) (> fnlwgt1 10.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (<= fnlwgt1 11.0) (<= hours_per_week1 36.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (<= fnlwgt1 11.0) (> hours_per_week1 36.0) (<= workclass1 0.0) (<= age1 1.0) (<= sex1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (<= fnlwgt1 11.0) (> hours_per_week1 36.0) (<= workclass1 0.0) (<= age1 1.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (<= fnlwgt1 11.0) (> hours_per_week1 36.0) (<= workclass1 0.0) (> age1 1.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (<= fnlwgt1 11.0) (> hours_per_week1 36.0) (> workclass1 0.0) (<= education1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (<= fnlwgt1 11.0) (> hours_per_week1 36.0) (> workclass1 0.0) (> education1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (<= workclass1 0.0) (<= occupation1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (<= workclass1 0.0) (> occupation1 0.0) (<= relationship1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (<= workclass1 0.0) (> occupation1 0.0) (> relationship1 2.0) (<= age1 2.0) (<= hours_per_week1 42.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (<= workclass1 0.0) (> occupation1 0.0) (> relationship1 2.0) (<= age1 2.0) (> hours_per_week1 42.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (<= workclass1 0.0) (> occupation1 0.0) (> relationship1 2.0) (> age1 2.0) (<= hours_per_week1 39.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (<= workclass1 0.0) (> occupation1 0.0) (> relationship1 2.0) (> age1 2.0) (> hours_per_week1 39.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (> workclass1 0.0) (<= hours_per_week1 47.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (<= native_country1 4.0) (> fnlwgt1 11.0) (> workclass1 0.0) (> hours_per_week1 47.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (> native_country1 4.0) (<= fnlwgt1 8.0) (<= capital_loss1 5.0) (<= education1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (> native_country1 4.0) (<= fnlwgt1 8.0) (<= capital_loss1 5.0) (> education1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (> native_country1 4.0) (<= fnlwgt1 8.0) (> capital_loss1 5.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (<= race1 1.0) (> native_country1 4.0) (> fnlwgt1 8.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (> race1 1.0) (<= age1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (<= relationship1 3.0) (> race1 1.0) (> age1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (<= native_country1 1.0) (<= hours_per_week1 45.0) (<= fnlwgt1 10.0) (<= occupation1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (<= native_country1 1.0) (<= hours_per_week1 45.0) (<= fnlwgt1 10.0) (> occupation1 0.0) (<= fnlwgt1 5.0) (<= age1 2.0) (<= fnlwgt1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (<= native_country1 1.0) (<= hours_per_week1 45.0) (<= fnlwgt1 10.0) (> occupation1 0.0) (<= fnlwgt1 5.0) (<= age1 2.0) (> fnlwgt1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (<= native_country1 1.0) (<= hours_per_week1 45.0) (<= fnlwgt1 10.0) (> occupation1 0.0) (<= fnlwgt1 5.0) (> age1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (<= native_country1 1.0) (<= hours_per_week1 45.0) (<= fnlwgt1 10.0) (> occupation1 0.0) (> fnlwgt1 5.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (<= native_country1 1.0) (<= hours_per_week1 45.0) (> fnlwgt1 10.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (<= native_country1 1.0) (> hours_per_week1 45.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (<= education1 4.0) (> native_country1 1.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (<= workclass1 1.0) (> relationship1 3.0) (> education1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (<= martial_status1 2.0) (> workclass1 1.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (> martial_status1 2.0) (<= capital_loss1 15.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (<= occupation1 1.0) (> martial_status1 2.0) (> capital_loss1 15.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (<= hours_per_week1 39.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (> hours_per_week1 39.0) (<= workclass1 2.0) (<= fnlwgt1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (> hours_per_week1 39.0) (<= workclass1 2.0) (> fnlwgt1 3.0) (<= sex1 0.0) (<= hours_per_week1 42.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (> hours_per_week1 39.0) (<= workclass1 2.0) (> fnlwgt1 3.0) (<= sex1 0.0) (> hours_per_week1 42.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (> hours_per_week1 39.0) (<= workclass1 2.0) (> fnlwgt1 3.0) (> sex1 0.0) (<= education1 1.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (> hours_per_week1 39.0) (<= workclass1 2.0) (> fnlwgt1 3.0) (> sex1 0.0) (> education1 1.0) (<= fnlwgt1 6.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (> hours_per_week1 39.0) (<= workclass1 2.0) (> fnlwgt1 3.0) (> sex1 0.0) (> education1 1.0) (> fnlwgt1 6.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (<= native_country1 4.0) (> hours_per_week1 39.0) (> workclass1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (<= race1 0.0) (> native_country1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (<= fnlwgt1 10.0) (> race1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (<= relationship1 2.0) (> fnlwgt1 10.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (<= hours_per_week1 42.0) (<= age1 6.0) (<= education1 1.0) (<= age1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (<= hours_per_week1 42.0) (<= age1 6.0) (<= education1 1.0) (> age1 3.0) (<= martial_status1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (<= hours_per_week1 42.0) (<= age1 6.0) (<= education1 1.0) (> age1 3.0) (> martial_status1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (<= hours_per_week1 42.0) (<= age1 6.0) (> education1 1.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (<= hours_per_week1 42.0) (> age1 6.0) (<= education1 8.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (<= hours_per_week1 42.0) (> age1 6.0) (> education1 8.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (> hours_per_week1 42.0) (<= age1 4.0) (<= education1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (> hours_per_week1 42.0) (<= age1 4.0) (> education1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (<= occupation1 2.0) (> relationship1 2.0) (> hours_per_week1 42.0) (> age1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (> occupation1 2.0) (<= hours_per_week1 44.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (> occupation1 2.0) (> hours_per_week1 44.0) (<= fnlwgt1 6.0) (<= occupation1 3.0) (<= sex1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (> occupation1 2.0) (> hours_per_week1 44.0) (<= fnlwgt1 6.0) (<= occupation1 3.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (> occupation1 2.0) (> hours_per_week1 44.0) (<= fnlwgt1 6.0) (> occupation1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (<= capital_loss1 3.0) (> occupation1 2.0) (> hours_per_week1 44.0) (> fnlwgt1 6.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (> capital_loss1 3.0) (<= native_country1 56.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (<= hours_per_week1 45.0) (> capital_loss1 3.0) (> native_country1 56.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (<= occupation1 3.0) (<= fnlwgt1 11.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (<= occupation1 3.0) (> fnlwgt1 11.0) (<= education1 4.0) (<= education1 2.0) (<= fnlwgt1 15.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (<= occupation1 3.0) (> fnlwgt1 11.0) (<= education1 4.0) (<= education1 2.0) (> fnlwgt1 15.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (<= occupation1 3.0) (> fnlwgt1 11.0) (<= education1 4.0) (> education1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (<= occupation1 3.0) (> fnlwgt1 11.0) (> education1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (> occupation1 3.0) (<= age1 3.0) (<= fnlwgt1 3.0) (<= sex1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (> occupation1 3.0) (<= age1 3.0) (<= fnlwgt1 3.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (> occupation1 3.0) (<= age1 3.0) (> fnlwgt1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (<= workclass1 0.0) (> occupation1 3.0) (> age1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (<= hours_per_week1 51.0) (> workclass1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (> hours_per_week1 51.0) (<= workclass1 1.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (> hours_per_week1 51.0) (> workclass1 1.0) (<= occupation1 3.0) (<= sex1 0.0) (<= hours_per_week1 67.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (> hours_per_week1 51.0) (> workclass1 1.0) (<= occupation1 3.0) (<= sex1 0.0) (> hours_per_week1 67.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (> hours_per_week1 51.0) (> workclass1 1.0) (<= occupation1 3.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (<= race1 3.0) (> hours_per_week1 51.0) (> workclass1 1.0) (> occupation1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (<= martial_status1 2.0) (> race1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (> martial_status1 2.0) (<= capital_loss1 9.0) (<= hours_per_week1 57.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (> martial_status1 2.0) (<= capital_loss1 9.0) (> hours_per_week1 57.0) (<= fnlwgt1 11.0) (<= fnlwgt1 4.0) (<= martial_status1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (> martial_status1 2.0) (<= capital_loss1 9.0) (> hours_per_week1 57.0) (<= fnlwgt1 11.0) (<= fnlwgt1 4.0) (> martial_status1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (> martial_status1 2.0) (<= capital_loss1 9.0) (> hours_per_week1 57.0) (<= fnlwgt1 11.0) (> fnlwgt1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (> martial_status1 2.0) (<= capital_loss1 9.0) (> hours_per_week1 57.0) (> fnlwgt1 11.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (<= native_country1 3.0) (> martial_status1 2.0) (> capital_loss1 9.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (<= occupation1 4.0) (> hours_per_week1 35.0) (> martial_status1 1.0) (> occupation1 1.0) (> hours_per_week1 45.0) (> native_country1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (<= fnlwgt1 5.0) (<= relationship1 3.0) (<= hours_per_week1 39.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (<= fnlwgt1 5.0) (<= relationship1 3.0) (> hours_per_week1 39.0) (<= workclass1 1.0) (<= education1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (<= fnlwgt1 5.0) (<= relationship1 3.0) (> hours_per_week1 39.0) (<= workclass1 1.0) (> education1 4.0) (<= sex1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (<= fnlwgt1 5.0) (<= relationship1 3.0) (> hours_per_week1 39.0) (<= workclass1 1.0) (> education1 4.0) (> sex1 0.0) (<= education1 11.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (<= fnlwgt1 5.0) (<= relationship1 3.0) (> hours_per_week1 39.0) (<= workclass1 1.0) (> education1 4.0) (> sex1 0.0) (> education1 11.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (<= fnlwgt1 5.0) (<= relationship1 3.0) (> hours_per_week1 39.0) (> workclass1 1.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (<= fnlwgt1 5.0) (> relationship1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (> fnlwgt1 5.0) (<= education1 0.0) (<= fnlwgt1 8.0) (<= workclass1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (> fnlwgt1 5.0) (<= education1 0.0) (<= fnlwgt1 8.0) (> workclass1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (> fnlwgt1 5.0) (<= education1 0.0) (> fnlwgt1 8.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (<= hours_per_week1 41.0) (> fnlwgt1 5.0) (> education1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (> hours_per_week1 41.0) (<= workclass1 2.0) (<= age1 3.0) (<= education1 2.0) (<= fnlwgt1 19.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (> hours_per_week1 41.0) (<= workclass1 2.0) (<= age1 3.0) (<= education1 2.0) (> fnlwgt1 19.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (> hours_per_week1 41.0) (<= workclass1 2.0) (<= age1 3.0) (> education1 2.0) (<= native_country1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (> hours_per_week1 41.0) (<= workclass1 2.0) (<= age1 3.0) (> education1 2.0) (> native_country1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (> hours_per_week1 41.0) (<= workclass1 2.0) (> age1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (<= occupation1 6.0) (> hours_per_week1 41.0) (> workclass1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (> occupation1 6.0) (<= hours_per_week1 49.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (> occupation1 6.0) (> hours_per_week1 49.0) (<= occupation1 7.0) (<= race1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (> occupation1 6.0) (> hours_per_week1 49.0) (<= occupation1 7.0) (> race1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (<= martial_status1 1.0) (> occupation1 6.0) (> hours_per_week1 49.0) (> occupation1 7.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (> martial_status1 1.0) (<= capital_gain1 0.0) (<= education1 12.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (> martial_status1 1.0) (<= capital_gain1 0.0) (> education1 12.0) (<= fnlwgt1 1.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (> martial_status1 1.0) (<= capital_gain1 0.0) (> education1 12.0) (> fnlwgt1 1.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (> martial_status1 1.0) (> capital_gain1 0.0) (<= race1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (<= hours_per_week1 54.0) (> martial_status1 1.0) (> capital_gain1 0.0) (> race1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (<= martial_status1 1.0) (<= hours_per_week1 55.0) (<= fnlwgt1 7.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (<= martial_status1 1.0) (<= hours_per_week1 55.0) (> fnlwgt1 7.0) (<= occupation1 9.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (<= martial_status1 1.0) (<= hours_per_week1 55.0) (> fnlwgt1 7.0) (> occupation1 9.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (<= martial_status1 1.0) (> hours_per_week1 55.0) (<= fnlwgt1 8.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (<= martial_status1 1.0) (> hours_per_week1 55.0) (> fnlwgt1 8.0) (<= hours_per_week1 67.0) (<= occupation1 9.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (<= martial_status1 1.0) (> hours_per_week1 55.0) (> fnlwgt1 8.0) (<= hours_per_week1 67.0) (> occupation1 9.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (<= martial_status1 1.0) (> hours_per_week1 55.0) (> fnlwgt1 8.0) (> hours_per_week1 67.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (<= occupation1 7.0) (<= hours_per_week1 58.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (<= occupation1 7.0) (> hours_per_week1 58.0) (<= native_country1 5.0) (<= workclass1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (<= occupation1 7.0) (> hours_per_week1 58.0) (<= native_country1 5.0) (> workclass1 0.0) (<= fnlwgt1 6.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (<= occupation1 7.0) (> hours_per_week1 58.0) (<= native_country1 5.0) (> workclass1 0.0) (> fnlwgt1 6.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (<= occupation1 7.0) (> hours_per_week1 58.0) (> native_country1 5.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (> occupation1 7.0) (<= hours_per_week1 82.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (> occupation1 7.0) (> hours_per_week1 82.0) (<= age1 4.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (<= race1 0.0) (> martial_status1 1.0) (> occupation1 7.0) (> hours_per_week1 82.0) (> age1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (<= workclass1 2.0) (> race1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (<= capital_loss1 5.0) (> hours_per_week1 54.0) (> workclass1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (> capital_loss1 5.0) (<= workclass1 23.0) (<= native_country1 23.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (> capital_loss1 5.0) (<= workclass1 23.0) (> native_country1 23.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (<= capital_gain1 1.0) (> occupation1 4.0) (> capital_loss1 5.0) (> workclass1 23.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (<= native_country1 6.0) (<= occupation1 10.0) (<= capital_gain1 2.0) (<= race1 2.0) (<= martial_status1 3.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (<= native_country1 6.0) (<= occupation1 10.0) (<= capital_gain1 2.0) (<= race1 2.0) (> martial_status1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (<= native_country1 6.0) (<= occupation1 10.0) (<= capital_gain1 2.0) (> race1 2.0) (<= occupation1 6.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (<= native_country1 6.0) (<= occupation1 10.0) (<= capital_gain1 2.0) (> race1 2.0) (> occupation1 6.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (<= native_country1 6.0) (<= occupation1 10.0) (> capital_gain1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (<= native_country1 6.0) (> occupation1 10.0) (<= capital_loss1 7.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (<= native_country1 6.0) (> occupation1 10.0) (> capital_loss1 7.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (<= capital_gain1 9.0) (<= capital_loss1 21.0) (<= age1 3.0) (<= fnlwgt1 14.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (<= capital_gain1 9.0) (<= capital_loss1 21.0) (<= age1 3.0) (> fnlwgt1 14.0) (<= native_country1 48.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (<= capital_gain1 9.0) (<= capital_loss1 21.0) (<= age1 3.0) (> fnlwgt1 14.0) (> native_country1 48.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (<= capital_gain1 9.0) (<= capital_loss1 21.0) (> age1 3.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (<= capital_gain1 9.0) (> capital_loss1 21.0) (<= education1 4.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (<= capital_gain1 9.0) (> capital_loss1 21.0) (> education1 4.0) (<= fnlwgt1 60.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (<= capital_gain1 9.0) (> capital_loss1 21.0) (> education1 4.0) (> fnlwgt1 60.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (> capital_gain1 9.0) (<= workclass1 31.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (> capital_gain1 9.0) (> workclass1 31.0) (<= capital_loss1 12.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (<= workclass1 36.0) (> native_country1 6.0) (> capital_gain1 9.0) (> workclass1 31.0) (> capital_loss1 12.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (<= capital_gain1 14.0) (<= occupation1 38.0) (<= hours_per_week1 62.0) (<= occupation1 36.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (<= capital_gain1 14.0) (<= occupation1 38.0) (<= hours_per_week1 62.0) (> occupation1 36.0) (<= education1 12.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (<= capital_gain1 14.0) (<= occupation1 38.0) (<= hours_per_week1 62.0) (> occupation1 36.0) (> education1 12.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (<= capital_gain1 14.0) (<= occupation1 38.0) (> hours_per_week1 62.0) (<= hours_per_week1 76.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (<= capital_gain1 14.0) (<= occupation1 38.0) (> hours_per_week1 62.0) (> hours_per_week1 76.0) (<= age1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (<= capital_gain1 14.0) (<= occupation1 38.0) (> hours_per_week1 62.0) (> hours_per_week1 76.0) (> age1 2.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (<= capital_gain1 14.0) (> occupation1 38.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (<= workclass1 57.0) (<= occupation1 70.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (<= workclass1 57.0) (> occupation1 70.0) (<= native_country1 51.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (<= workclass1 57.0) (> occupation1 70.0) (> native_country1 51.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (> workclass1 57.0) (<= occupation1 49.0) (<= education1 10.0) (<= education1 2.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (> workclass1 57.0) (<= occupation1 49.0) (<= education1 10.0) (> education1 2.0) (<= occupation1 16.0) (<= sex1 0.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (> workclass1 57.0) (<= occupation1 49.0) (<= education1 10.0) (> education1 2.0) (<= occupation1 16.0) (> sex1 0.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (> workclass1 57.0) (<= occupation1 49.0) (<= education1 10.0) (> education1 2.0) (> occupation1 16.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (> workclass1 57.0) (<= occupation1 49.0) (> education1 10.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (<= capital_loss1 34.0) (> workclass1 57.0) (> occupation1 49.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (<= capital_gain1 24.0) (> workclass1 36.0) (> capital_gain1 14.0) (> capital_loss1 34.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (> capital_gain1 24.0) (<= capital_gain1 32.0) (<= native_country1 63.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (> capital_gain1 24.0) (<= capital_gain1 32.0) (> native_country1 63.0) (<= workclass1 62.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (> capital_gain1 24.0) (<= capital_gain1 32.0) (> native_country1 63.0) (> workclass1 62.0) (<= occupation1 55.0) ) (= Class1 1)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (> capital_gain1 24.0) (<= capital_gain1 32.0) (> native_country1 63.0) (> workclass1 62.0) (> occupation1 55.0) ) (= Class1 0)))
(assert (=> (and (> martial_status1 0.0) (> capital_gain1 1.0) (> capital_gain1 24.0) (> capital_gain1 32.0) ) (= Class1 1)))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))



(assert (=  age0 age1))



(assert (=  workclass0 workclass1))



(assert (=  fnlwgt0 fnlwgt1))



(assert (=  education0 education1))



(assert (=  martial_status0 martial_status1))



(assert (=  occupation0 occupation1))



(assert (=  relationship0 relationship1))



(assert (=  race0 race1))



(assert (not(= sex0 sex1)))



(assert (=  capital_gain0 capital_gain1))



(assert (=  capital_loss0 capital_loss1))



(assert (=  hours_per_week0 hours_per_week1))



(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))

(assert (=  age0 age1))

(assert (=  workclass0 workclass1))

(assert (=  fnlwgt0 fnlwgt1))

(assert (=  education0 education1))

(assert (=  martial_status0 martial_status1))

(assert (=  occupation0 occupation1))

(assert (=  relationship0 relationship1))

(assert (=  race0 race1))

(assert (not(= sex0 sex1)))

(assert (=  capital_gain0 capital_gain1))

(assert (=  capital_loss0 capital_loss1))

(assert (=  hours_per_week0 hours_per_week1))

(assert (=  native_country0 native_country1))


(assert(not (= Class0 Class1)))
(assert(not (= Class0 Class1)))(assert(not (=  Class0 Class1)))
(assert(not (=  Class0 Class1)))


(assert (not (> capital_gain1 33.5)))
(check-sat) 
(get-model) 
