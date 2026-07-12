window.LAB_DATA = window.LAB_DATA || {};
window.LAB_DATA["quasi"] = {
 "parallel": {
  "periods": [
   0,
   1,
   2,
   3,
   4,
   5,
   6,
   7,
   8,
   9,
   10,
   11,
   12,
   13,
   14,
   15
  ],
  "treated_mean": [
   3.211,
   3.717,
   4.497,
   5.095,
   5.298,
   6.236,
   6.313,
   7.042,
   9.542,
   9.922,
   10.581,
   10.954,
   11.148,
   12.003,
   12.053,
   12.492
  ],
  "control_mean": [
   -0.053,
   0.52,
   0.837,
   1.469,
   2.145,
   2.788,
   3.13,
   3.605,
   3.79,
   4.248,
   4.907,
   5.778,
   5.99,
   6.877,
   7.145,
   7.387
  ],
  "estimates": {
   "post_vs_pre": 5.911,
   "treated_vs_control": 5.322,
   "difference_in_differences": 1.951
  },
  "expected": {
   "post_vs_pre": 6.0,
   "treated_vs_control": 5.292,
   "difference_in_differences": 2.0
  },
  "placebo_did": -0.132
 },
 "violated": {
  "periods": [
   0,
   1,
   2,
   3,
   4,
   5,
   6,
   7,
   8,
   9,
   10,
   11,
   12,
   13,
   14,
   15
  ],
  "treated_mean": [
   5.27,
   6.071,
   7.028,
   7.888,
   8.364,
   9.272,
   10.054,
   11.107,
   13.717,
   14.577,
   15.537,
   16.156,
   16.821,
   17.875,
   18.636,
   19.452
  ],
  "control_mean": [
   0.254,
   0.81,
   1.192,
   1.635,
   2.281,
   2.875,
   3.191,
   3.816,
   4.493,
   4.75,
   5.397,
   5.967,
   6.305,
   6.879,
   7.389,
   7.692
  ],
  "estimates": {
   "post_vs_pre": 8.465,
   "treated_vs_control": 10.487,
   "difference_in_differences": 4.363
  },
  "expected": {
   "post_vs_pre": 8.4,
   "treated_vs_control": 10.552,
   "difference_in_differences": 4.4
  },
  "placebo_did": 1.067
 },
 "true_effect": 2.0,
 "launch_period": 8,
 "differential_trend": 0.3
};
