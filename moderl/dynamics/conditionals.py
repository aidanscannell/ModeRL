from typing import Optional

import tensorflow as tf


def svgp_covariance_conditional(X1, X2, svgp):
    return covariance_conditional(
        X1,
        X2,
        kernel=svgp.kernel,
        inducing_variable=svgp.inducing_variable,
        f=svgp.q_mu,
        q_sqrt=svgp.q_sqrt,
        white=svgp.whiten,
    )


def covariance_conditional(
    X1, X2, kernel, inducing_variable, f, q_sqrt=None, white=False
):
    K12 = kernel(X1, X2)
    # print("K12")
    # print(K12.shape)
    Kmm = kernel(inducing_variable.Z, inducing_variable.Z)
    # print("Kmm")
    # print(Kmm.shape)
    Lm = tf.linalg.cholesky(Kmm)
    # print("Lm")
    # print(Lm.shape)

    Km1 = kernel(inducing_variable.Z, X1)
    Km2 = kernel(inducing_variable.Z, X2)
    # print("Km1.shape")
    # print(Km1.shape)
    # print(Km2.shape)
    return base_covariance_conditional(
        Km1=Km1,
        Km2=Km2,
        Lm=Lm,
        K12=K12,
        f=f,
        q_sqrt=q_sqrt,
        white=white,
        # X1,
        # X2,
        # kernel=svgp.kernel,
        # inducing_variable=svgp.inducing_variable,
        # q_mu=svgp.q_mu,
        # q_sqrt=svgp.q_sqrt,
    )


def base_covariance_conditional(
    Km1: tf.Tensor,
    Km2: tf.Tensor,
    Lm: tf.Tensor,
    K12: tf.Tensor,
    f: tf.Tensor,
    *,
    # full_cov=False,
    q_sqrt: Optional[tf.Tensor] = None,
    white=False,
):
    # compute kernel stuff
    num_func = tf.shape(f)[-1]  # R
    N1 = tf.shape(Km1)[-1]
    N2 = tf.shape(Km2)[-1]
    M = tf.shape(f)[-2]

    # get the leading dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K1 = tf.rank(Km1)
    K2 = tf.rank(Km2)
    perm_1 = tf.concat(
        [
            tf.reshape(tf.range(1, K1 - 1), [K1 - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K1 - 1, [1]),
        ],
        0,
    )  # [N]
    perm_2 = tf.concat(
        [
            tf.reshape(tf.range(1, K2 - 1), [K2 - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K2 - 1, [1]),
        ],
        0,
    )  # [N]
    Km1 = tf.transpose(Km1, perm_1)  # [..., M, N1]
    Km2 = tf.transpose(Km2, perm_2)  # [..., M, N2]
    # print("Km1")
    # print(Km1.shape)
    # print(Km2.shape)

    shape_constraints = [
        (Km1, [..., "M", "N1"]),
        (Km2, [..., "M", "N2"]),
        (Lm, ["M", "M"]),
        (K12, [..., "N1", "N2"]),
        (f, ["M", "R"]),
    ]
    if q_sqrt is not None:
        shape_constraints.append(
            (q_sqrt, (["M", "R"] if q_sqrt.shape.ndims == 2 else ["R", "M", "M"]))
        )
    tf.debugging.assert_shapes(
        shape_constraints,
        message="base_conditional() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    leading_dims = tf.shape(Km1)[:-2]

    # Compute the projection matrix A
    Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., M, M]
    # print("Lm")
    # print(Lm.shape)
    A1 = tf.linalg.triangular_solve(Lm, Km1, lower=True)  # [..., M, N1]
    # print("A1")
    # print(A1.shape)
    A2 = tf.linalg.triangular_solve(Lm, Km2, lower=True)  # [..., M, N2]
    # print("A2")
    # print(A2.shape)

    # compute the covariance due to the conditioning
    fcov = K12 - tf.linalg.matmul(A1, A2, transpose_a=True)  # [..., N1, N2]
    # if not full_cov:
    #     fcov = tf.linalg.diag_part(fcov)
    cov_shape = tf.concat([leading_dims, [num_func, N1, N2]], 0)
    fcov = tf.broadcast_to(tf.expand_dims(fcov, -3), cov_shape)  # [..., R, N1, N2]

    # another backsubstitution in the unwhitened case
    # TODO what if white=False?
    # if not white:
    #     A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)

    if q_sqrt is not None:
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA1 = A1 * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N1]
            LTA2 = A2 * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N2]
        elif q_sqrt_dims == 3:
            L = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
            L_shape = tf.shape(L)
            L = tf.broadcast_to(L, tf.concat([leading_dims, L_shape], 0))

            shape1 = tf.concat([leading_dims, [num_func, M, N1]], axis=0)
            shape2 = tf.concat([leading_dims, [num_func, M, N2]], axis=0)
            A1_tiled = tf.broadcast_to(tf.expand_dims(A1, -3), shape1)
            A2_tiled = tf.broadcast_to(tf.expand_dims(A2, -3), shape2)
            LTA1 = tf.linalg.matmul(L, A1_tiled, transpose_a=True)  # [R, M, N1]
            LTA2 = tf.linalg.matmul(L, A2_tiled, transpose_a=True)  # [R, M, N2]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))
        # print("LTA1")
        # print(LTA1.shape)
        # print(LTA2.shape)

        # fcov = fcov + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
        fcov = fcov + tf.linalg.matmul(LTA1, LTA2, transpose_a=True)  # [R, N1, N2]

    # if not full_cov:
    #     fcov = tf.linalg.adjoint(fcov)  # [N, R]

    # shape_constraints = [
    #     (Km1, [..., "M", "N1"]),  # tensor included again for N dimension
    #     (f, [..., "M", "R"]),  # tensor included again for R dimension
    #     (fmean, [..., "N", "R"]),
    #     (fvar, [..., "R", "N1", "N2"] if full_cov else [..., "N1", "R"]),
    # ]
    # tf.debugging.assert_shapes(
    #     shape_constraints, message="base_conditional() return values"
    # )

    # return fmean, fvar
    return fcov


# def base_svgp_conditional(X1, X2, kernel, inducing_variable, q_mu, q_sqrt):
#     K12 = kernel(X1, X2)
#     # print("K12")
#     # print(K12.shape)
#     Kzz = kernel(inducing_variable.Z, inducing_variable.Z)
#     # jitter=1e-6
#     jitter = 1e-4
#     Kzz += jitter * tf.eye(inducing_variable.num_inducing, dtype=Kzz.dtype)
#     # print("Kzz")
#     # print(Kzz.shape)
#     Lz = tf.linalg.cholesky(Kzz)
#     # print("Lz")
#     # print(Lz.shape)

#     K1z = kernel(X1, inducing_variable.Z)
#     # Kz2 = kernel(X2, inducing_variable.Z)
#     Kz2 = kernel(inducing_variable.Z, X2)
#     # print("K1z.shape")
#     # print(K1z.shape)
#     # print(Kz2.shape)

#     S = q_sqrt @ tf.transpose(q_sqrt, [0, 2, 1])
#     A = Kzz - S[0, :, :]
#     # print("A.shape")
#     # print(A.shape)
#     # B = Kz1 @ A @ tf.transpose(Kz2)
#     B = K1z @ A @ Kz2
#     # print("B.shape")
#     # print(B.shape)
#     K = K12 - B
#     # print("K.shape")
#     # print(K.shape)
#     return K
