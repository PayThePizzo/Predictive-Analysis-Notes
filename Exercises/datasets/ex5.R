## run the codebelow to create the ex5 data.frame
# alternatively, save this in your working directory and run
# source("ex5.R")
ex5 <-   structure(list(X = c(11.606, 12.174, 13.405, 11.769, 10.406, 
                              11.075, 12.056, 11.039, 14.719, 10.392, 10.08, 10.843, 10.735, 
                              12.87, 11.06, 11.914, 9.608, 10.131, 9.1129, 9.8603, 11.605, 
                              10.823, 11.357, 9.8128, 10.683, 12.066, 12.424, 10.371, 11.751, 
                              12.439, 11.884, 12.749, 10.685, 11.345, 10.238, 11.127, 11.095, 
                              9.9815, 10.958, 11.674, 9.8715, 11.779, 12.447, 11.211, 9.5583, 
                              10.269, 11.513, 9.988, 12.293, 12.478, 11.037, 11.296, 11.833, 
                              12.999, 10.087, 12.139, 10.354, 11.091, 11.861, 10.857, 10.72, 
                              11.484, 10.313, 10.126, 9.3146, 10.878, 9.4501, 11.266, 9.9204, 
                              9.0944, 11.585, 11.739, 11.639, 9.8402, 11.993, 10.673, 10.453, 
                              12.194, 9.8571, 11.507, 12.374, 10.309, 11.656, 9.7955, 10.69, 
                              10.934, 12.595, 10.828, 12.036, 12.43, 9.9736, 13.274, 9.726, 
                              10.679, 10.78, 10.665, 10.837, 12.081, 8.605, 11.92, 9.131, 6.4008, 
                              10.451, 7.7125, 9.4625, 7.2924, 8.9559, 9.1629, 8.5554, 8.2556, 
                              9.8577, 8.2367, 8.3237, 7.2136, 9.5507, 10.516, 8.4973, 7.2663, 
                              8.3688, 9.3085, 9.8444, 10.67, 7.3194, 7.4268, 9.4135, 8.9752, 
                              8.9973, 8.4082, 8.6107, 11.019, 8.9282, 8.4122, 8.5313, 8.8529, 
                              8.6348, 10.297, 9.1308, 9.2968, 10.63, 9.1285, 8.7458, 8.1152, 
                              8.964, 10.08, 8.7707, 8.4855, 8.8832, 7.9985, 9.3724, 8.7502, 
                              9.415, 8.9924, 8.5221, 8.4093, 6.3979, 9.1085, 7.1046, 9.515, 
                              10.645, 7.4724, 11.004, 8.1637, 8.8085, 8.6564, 8.4256, 6.7608, 
                              9.0478, 8.5242, 8.2468, 7.7033, 10.013, 9.4715, 9.4656, 8.3038, 
                              9.2847, 10.235, 9.0841, 9.3958, 8.1439, 10.556, 8.7568, 7.0247, 
                              8.7388, 7.047, 10.543, 10.269, 11.359, 10.108, 6.975, 7.448, 
                              10.21, 7.734, 9.5718, 8.5883, 9.8239, 8.1825, 8.9061, 7.2438, 
                              9.7674, 9.2498, 5.6991, 5.994, 4.6517, 6.5379, 7.7404, 8.8833, 
                              6.5273, 7.9228, 8.6579, 7.6123, 7.4977, 7.9968, 7.4491, 6.1009, 
                              4.8876, 5.9108, 7.3018, 5.117, 9.8178, 5.7129, 6.7137, 6.6559, 
                              7.4847, 8.0451, 7.2918, 7.7211, 6.495, 7.9939, 5.1811, 8.4697, 
                              4.7941, 5.1016, 7.304, 6.4315, 7.7593, 7.786, 5.7017, 7.5189, 
                              8.1013, 6.3627, 9.7579, 3.8281, 8.9888, 5.5938, 7.6891, 6.2417, 
                              6.0489, 7.4118, 7.3371, 7.1172, 7.181, 8.0971, 6.4431, 7.3119, 
                              9.181, 7.0473, 6.4147, 7.3615, 8.9839, 8.9613, 7.8378, 6.7546, 
                              6.7006, 5.7046, 6.6364, 6.5317, 7.1354, 8.6221, 8.8552, 6.8592, 
                              8.4876, 7.4162, 7.1897, 6.0338, 9.103, 8.4701, 8.22, 8.8928, 
                              10.1, 5.0054, 10.072, 6.6856, 6.9938, 5.7, 7.4328, 7.7755, 7.195, 
                              7.2175, 9.1698, 5.5384, 6.4681, 7.2249, 7.7418, 8.2652, 6.9946, 
                              6.9545, 7.186, 8.5464, 8.0661, 9.0229, 4.7073, 5.8839, 4.641, 
                              7.9008, 4.7637, 4.8773, 4.7924, 5.7245, 3.8338, 4.9473, 6.0522, 
                              6.2167, 4.5251, 5.5644, 6.1483, 6.0794, 5.7245, 3.9963, 1.4072, 
                              5.3958, 6.7242, 4.542, 3.5422, 4.656, 4.3561, 3.505, 3.8548, 
                              5.5731, 6.1108, 5.2197, 5.2172, 4.7553, 4.9577, 6.0036, 4.4051, 
                              3.7303, 3.0661, 4.9695, 3.1073, 4.0844, 6.0298, 6.1912, 3.1423, 
                              6.0891, 6.2816, 4.5285, 3.2964, 5.949, 6.9267, 5.2735, 5.3179, 
                              5.5927, 5.5848, 6.1658, 4.0554, 4.2989, 7.6037, 5.2675, 5.1419, 
                              3.6473, 4.2022, 5.5793, 6.9267, 3.6534, 5.7448, 5.6413, 5.8976, 
                              3.2904, 6.143, 4.0502, 4.109, 3.6119, 6.0867, 4.8727, 5.9531, 
                              4.562, 3.1241, 4.5621, 3.7676, 7.4156, 7.1933, 5.9927, 5.3338, 
                              6.2996, 5.7557, 3.7286, 4.4258, 4.4647, 2.8295, 6.5903, 5.5797, 
                              6.0117, 4.7913, 4.8508, 5.0491, 4.8225, 4.0617, 4.4834, 2.7858, 
                              6.7048, 3.2172, 4.084, 4.9708, 3.2559, 2.878, 3.001, 2.6274, 
                              3.3649, 1.8761, 1.3461, 0.99084, 3.7321, 3.8159, 3.2716, 4.2505, 
                              0.96373, 4.8077, 3.102, 3.8946, 4.5382, 3.1789, 0.022579, 3.4586, 
                              2.62, 2.0452, 3.51, 3.201, 4.5934, 3.9961, 3.6033, 1.476, 4.5198, 
                              2.8212, 3.6406, 3.5147, 2.8417, 4.0657, 2.2762, 3.0169, 4.315, 
                              3.3498, 2.4275, 2.5621, 2.5799, 3.78, 1.9604, 2.0607, 3.8666, 
                              3.5985, 2.0881, 4.492, 2.99, 1.2858, 4.9377, 3.8596, 3.6822, 
                              3.1077, 2.698, 2.5984, 4.5529, 1.5816, 3.8129, 4.6495, 2.0125, 
                              4.1426, 1.7109, 4.6617, 5.1923, 4.2757, 2.8008, 3.7094, 3.1188, 
                              2.5383, 2.2783, 4.7852, 4.0065, 2.3068, 2.9354, 3.1969, 1.2978, 
                              4.3865, 3.766, 2.4942, 5.1024, 0.95397, 2.9959, 2.3573, 2.4649, 
                              2.066, 3.1673, 2.2995, 3.6921, 2.5857, 2.4542, 3.4519, 3.446, 
                              3.6884, 3.1446, 4.118, 4.7846), 
                        Y = c(2.6457, 4.2192, 5.6656,
                              3.2294, 1.6661, 1.7582, 1.8237, 3.914, 6.517, 2.5836, 1.922, 
                              1.88, 1.7027, 3.7778, 2.5508, 3.9872, -0.70366, 2.5084, 1.0821, 
                              2.9825, 2.0342, 2.6032, 3.1326, 2.9349, 2.8369, 3.2781, 2.9753, 
                              2.0636, 3.6271, 5.0323, 2.8991, 4.9632, 2.7817, 2.8336, 1.5224, 
                              4.1091, 3.04, 2.2005, 2.4157, 4.454, 1.215, 2.9192, 4.1891, 2.5889, 
                              2.8947, 2.4225, 3.2499, 2.9566, 4.6385, 2.0474, 2.629, 3.8397, 
                              3.3588, 3.9723, 0.94363, 4.0698, 3.5073, 2.6232, 2.7098, 3.5004, 
                              1.9335, 2.9715, 2.314, 2.3113, 2.4277, 2.3549, 1.4753, 2.9094, 
                              2.9146, 2.0783, 3.4957, 4.0842, 4.3481, 2.0697, 4.6093, 4.5125, 
                              2.5101, 2.9548, 2.3859, 2.1802, 4.8427, 2.4661, 4.146, 2.5078, 
                              2.0881, 3.6746, 4.6142, 2.8592, 4.8541, 4.6921, 3.1491, 4.6747, 
                              2.352, 1.5993, 3.8134, 4.2335, 3.5043, 2.9062, 1.5209, 2.9117, 
                              6.4118, 3.671, 6.3051, 3.259, 5.4141, 3.4541, 4.2078, 5.2308, 
                              5.2035, 4.2638, 6.0499, 4.4018, 4.0126, 3.4115, 5.2503, 6.8171, 
                              3.8643, 5.3015, 5.4896, 6.5074, 5.9163, 6.2398, 2.3443, 3.5344, 
                              4.6312, 4.8167, 6.0765, 6.1129, 3.7678, 6.4572, 5.0462, 3.56, 
                              5.5298, 3.8525, 4.9763, 6.3952, 4.939, 3.129, 7.7306, 2.3674, 
                              5.6133, 5.6097, 4.6343, 5.0044, 4.5353, 4.6818, 5.0572, 5.126, 
                              5.518, 3.4821, 6.0379, 5.6206, 4.1155, 3.4854, 2.4213, 5.5074, 
                              4.4432, 6.2265, 6.1663, 4.9233, 6.7028, 5.5353, 4.7674, 6.0446, 
                              3.6714, 3.255, 4.1935, 4.3744, 4.1576, 2.7265, 5.884, 4.464, 
                              4.5482, 4.418, 4.1378, 5.7137, 4.807, 4.7195, 4.3611, 6.4289, 
                              4.3984, 2.8476, 5.1696, 3.5838, 6.4691, 5.7227, 6.3172, 4.9116, 
                              2.698, 2.7831, 7.2119, 4.7972, 5.7317, 5.7206, 5.7903, 2.58, 
                              5.7662, 3.8666, 5.1974, 5.6659, 4.5725, 5.6433, 5.6693, 5.6825, 
                              7.9297, 9.5502, 6.8799, 8.0995, 7.8075, 7.5132, 7.6429, 7.7147, 
                              7.0612, 6.6592, 5.6988, 5.5952, 9.0274, 4.8965, 9.5446, 5.4401, 
                              6.9674, 6.5495, 9.1435, 7.6322, 6.5234, 7.147, 7.8745, 7.1784, 
                              4.9477, 7.3448, 5.9649, 6.021, 7.8535, 6.7841, 8.0034, 7.582, 
                              6.8497, 6.48, 8.5809, 5.7543, 8.213, 3.9196, 8.9344, 4.9611, 
                              8.1725, 7.0786, 5.9831, 6.4737, 6.6798, 7.6545, 8.247, 8.5158, 
                              7.1279, 7.9245, 8.6413, 7.1537, 7.0325, 8.0879, 8.7282, 7.6362, 
                              8.5304, 7.8757, 6.0725, 6.5798, 6.7157, 6.6508, 7.1373, 8.8807, 
                              9.1125, 7.825, 8.1643, 6.9641, 6.9551, 6.1741, 9.125, 6.6707, 
                              6.3444, 8.7083, 9.5324, 4.8973, 9.5498, 5.9734, 7.6295, 4.1499, 
                              5.9741, 6.0855, 6.8441, 8.8842, 7.9213, 5.5243, 6.578, 7.1485, 
                              7.1809, 7.2211, 6.9786, 7.0919, 6.0585, 9.5442, 9.2105, 8.1539, 
                              9.6891, 9.8798, 7.959, 11.002, 8.8103, 10.085, 9.4632, 9.2473, 
                              6.9941, 7.5801, 9.729, 10.814, 8.206, 9.4344, 8.8814, 8.4164, 
                              8.9604, 7.3332, 5.4086, 8.464, 10.688, 8.7296, 6.0677, 9.2725, 
                              8.4358, 9.0266, 8.8693, 9.1067, 10.881, 8.8111, 9.175, 8.463, 
                              8.7705, 9.9226, 8.3902, 7.7603, 7.3124, 7.8113, 7.4839, 8.2034, 
                              10.433, 9.4325, 8.5891, 8.8146, 10.947, 8.5686, 6.3153, 8.6518, 
                              10.646, 8.9887, 8.5862, 9.5844, 9.7833, 8.9491, 8.9787, 8.4189, 
                              11.431, 8.8684, 8.5608, 8.0238, 8.8797, 8.8276, 10, 7.2713, 10.012, 
                              9.5034, 9.862, 6.6951, 10.477, 7.6291, 7.846, 7.3799, 8.6915, 
                              8.0169, 9.6933, 8.4634, 8.4509, 7.3979, 7.2169, 10.851, 9.7784, 
                              9.9489, 8.7876, 9.4812, 9.5607, 7.192, 8.1695, 8.0119, 7.3504, 
                              9.1113, 9.8745, 10.471, 7.4898, 9.0705, 8.7015, 7.3817, 8.3669, 
                              9.0679, 7.141, 10.455, 10.189, 10.784, 11.949, 11.083, 10.518, 
                              10.394, 10.946, 11.341, 9.4792, 8.811, 8.6911, 11.451, 12.26, 
                              10.084, 13.753, 11.352, 11.919, 10.861, 11.176, 12.26, 9.4539, 
                              9.188, 11.757, 9.8866, 10.087, 10.295, 10.695, 11.297, 12.975, 
                              11.769, 9.0976, 12.614, 10.714, 12.012, 11.324, 10.277, 11.153, 
                              10.405, 12.805, 12.171, 11.012, 12.103, 11.367, 11.081, 11.543, 
                              11.411, 9.9349, 11.113, 11.583, 11.14, 11.599, 9.603, 9.7241, 
                              12.461, 13.016, 12.269, 10.433, 10.992, 10.384, 9.8421, 10.631, 
                              12.293, 11.903, 8.3349, 11.863, 10.719, 12.278, 13.075, 11.491, 
                              10.342, 10.384, 9.4857, 9.6797, 10.589, 12.978, 11.687, 10.901, 
                              12.098, 9.967, 9.9478, 13.192, 11.344, 11.13, 13.671, 9.2289, 
                              9.4616, 10.428, 9.2842, 10.118, 10.722, 12.557, 11.617, 10.806, 
                              12.062, 10.802, 11.113, 10.077, 10.264, 10.409, 12.249), 
                        Z = c("1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", 
                              "1", "1", "1", "1", "1", "1", "1", "1", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", 
                              "2", "2", "2", "2", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", "4", 
                              "4", "4", "4", "4", "4", "4", "4", "4", "4", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", 
                              "5", "5", "5", "5", "5")), row.names = c(NA, -500L), class = "data.frame")
