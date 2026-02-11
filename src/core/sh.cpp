#include "core/sh.hpp"
#include "core/gaussian.hpp"

#include <stdexcept>

namespace cugs {

torch::Tensor evaluate_sh_cpu(int degree,
                              const torch::Tensor& sh_coeffs,
                              const torch::Tensor& directions) {
    TORCH_CHECK(degree >= 0 && degree <= kMaxSHDegree,
                "SH degree must be 0..3, got ", degree);
    TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(1) == 3,
                "sh_coeffs must be [N, 3, C]");
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
                "directions must be [N, 3]");
    TORCH_CHECK(sh_coeffs.size(0) == directions.size(0),
                "Batch size mismatch");

    const int64_t n = sh_coeffs.size(0);
    const int required_coeffs = sh_coeff_count(degree);
    TORCH_CHECK(sh_coeffs.size(2) >= required_coeffs,
                "sh_coeffs has ", sh_coeffs.size(2), " coefficients but degree ",
                degree, " requires ", required_coeffs);

    // Ensure float32, contiguous, CPU
    auto coeffs = sh_coeffs.to(torch::kFloat32).contiguous();
    auto dirs = directions.to(torch::kFloat32).contiguous();

    auto result = torch::zeros({n, 3}, torch::kFloat32);

    auto c_acc = coeffs.accessor<float, 3>();   // [N, 3, C]
    auto d_acc = dirs.accessor<float, 2>();      // [N, 3]
    auto r_acc = result.accessor<float, 2>();    // [N, 3]

    for (int64_t i = 0; i < n; ++i) {
        const float x = d_acc[i][0];
        const float y = d_acc[i][1];
        const float z = d_acc[i][2];

        for (int ch = 0; ch < 3; ++ch) {
            float color = 0.0f;

            // Degree 0: Y_0^0 = C0
            color += kSH_C0 * c_acc[i][ch][0];

            if (degree >= 1) {
                // Degree 1: Y_1^{-1} = C1*y, Y_1^0 = C1*z, Y_1^1 = C1*x
                color += kSH_C1 * (
                    -c_acc[i][ch][1] * y +
                     c_acc[i][ch][2] * z +
                    -c_acc[i][ch][3] * x
                );
            }

            if (degree >= 2) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;

                // Degree 2 basis functions
                color += kSH_C2_0 * c_acc[i][ch][4] * xy;           // Y_2^{-2}
                color += kSH_C2_1 * c_acc[i][ch][5] * yz;           // Y_2^{-1}
                color += kSH_C2_2 * c_acc[i][ch][6] * (2*zz - xx - yy); // Y_2^0
                color += kSH_C2_3 * c_acc[i][ch][7] * xz;           // Y_2^1
                color += kSH_C2_4 * c_acc[i][ch][8] * (xx - yy);    // Y_2^2
            }

            if (degree >= 3) {
                const float xx = x * x, yy = y * y, zz = z * z;

                // Degree 3 basis functions
                color += kSH_C3_0 * c_acc[i][ch][9]  * y * (3*xx - yy);        // Y_3^{-3}
                color += kSH_C3_1 * c_acc[i][ch][10] * x * y * z;              // Y_3^{-2}
                color += kSH_C3_2 * c_acc[i][ch][11] * y * (4*zz - xx - yy);   // Y_3^{-1}
                color += kSH_C3_3 * c_acc[i][ch][12] * z * (2*zz - 3*xx - 3*yy); // Y_3^0
                color += kSH_C3_4 * c_acc[i][ch][13] * x * (4*zz - xx - yy);   // Y_3^1
                color += kSH_C3_5 * c_acc[i][ch][14] * z * (xx - yy);          // Y_3^2
                color += kSH_C3_6 * c_acc[i][ch][15] * x * (xx - 3*yy);        // Y_3^3
            }

            // Add the 0.5 bias (SH convention: color = SH_eval + 0.5)
            r_acc[i][ch] = color + 0.5f;
        }
    }

    return result;
}

} // namespace cugs
