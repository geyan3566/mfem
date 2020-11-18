#include "mfem.hpp"
#include "../../general/forall.hpp"
#include "genericintegrator.hpp"

#pragma once

namespace mfem
{
template<typename T>
struct supported_type
{
   static constexpr bool value = false;
};

template<>
struct supported_type<ParMesh>
{
   static constexpr bool value = true;
};

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
class QFunctionIntegrator : public GenericIntegrator
{
public:
   QFunctionIntegrator(qfunc_type f,
                       qfunc_grad_type f_grad,
                       qfunc_args_type const &... fargs);

   void Setup(const FiniteElementSpace &fes) override;

   void Apply(const Vector &, Vector &) const override;

   // y += F'(x) * v
   void ApplyGradient(const Vector &x,
                      const Vector &v,
                      Vector &y) const override;

protected:
   void Apply2D(const Vector &u_in_, Vector &y_) const;

   void ApplyGradient2D(const Vector &u_in_,
                        const Vector &v_in_,
                        Vector &y_) const;

   auto EvaluateFargValue(const Mesh& m) const;

   const FiniteElementSpace *fespace;
   const DofToQuad *maps;        ///< Not owned
   const GeometricFactors *geom; ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

   // Geometric factors
   Vector J_;
   Vector W_;

   qfunc_type qf;
   qfunc_grad_type qf_grad;
   std::tuple<qfunc_args_type...> qf_farg_values;
};

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::
   QFunctionIntegrator(qfunc_type f,
                       qfunc_grad_type df,
                       qfunc_args_type const &... fargs)
   : GenericIntegrator(nullptr), maps(nullptr), geom(nullptr), qf(f),
     qf_grad(df), qf_farg_values(std::tuple{fargs...})
{
   static_assert((supported_type<qfunc_args_type>::value && ...),
                 "Type not supported for parameter expansion. See "
                 "documentation for supported types.");
}

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::Setup(
   const FiniteElementSpace &fes)
{
   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   if (mesh->GetNE() == 0)
   {
      return;
   }
   const FiniteElement &el = *fes.GetFE(0);
   ElementTransformation *T = mesh->GetElementTransformation(0);
   const IntegrationRule *ir = nullptr;
   if (!IntRule)
   {
      IntRule = &IntRules.Get(el.GetGeomType(), el.GetOrder() * 2);
   }
   ir = IntRule;

   dim = mesh->Dimension();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir,
                                    GeometricFactors::COORDINATES
                                       | GeometricFactors::JACOBIANS);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   //    pa_data.SetSize(ne * nq, Device::GetDeviceMemoryType());

   W_.SetSize(nq, Device::GetDeviceMemoryType());
   W_.GetMemory().CopyFrom(ir->GetWeights().GetMemory(), nq);

   // J.SetSize(ne * nq, Device::GetDeviceMemoryType());
   J_ = geom->J;
}

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
auto QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::
   EvaluateFargValue(const Mesh &m) const
{
   return std::tuple{};
}

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::Apply(
   const Vector &x, Vector &y) const
{
   Apply2D(x, y);
}

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::Apply2D(
   const Vector &u_in_, Vector &y_) const
{
   int D1D = dofs1D;
   int Q1D = quad1D;
   int NE = ne;

   auto v1d = Reshape(maps->B.Read(), Q1D, D1D);
   auto dv1d_dX = Reshape(maps->G.Read(), Q1D, D1D);
   // (NQ x SDIM x DIM x NE)
   auto J = Reshape(J_.Read(), Q1D, Q1D, 2, 2, NE);
   auto W = Reshape(W_.Read(), Q1D, Q1D);
   auto u = Reshape(u_in_.Read(), D1D, D1D, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   // MFEM_FORALL(e, NE, {
   for (int e = 0; e < NE; e++)
   {
      // loop over quadrature points
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            double u_q = 0.0;
            double du_dX_q[2] = {0.0};
            for (int ix = 0; ix < D1D; ix++)
            {
               for (int iy = 0; iy < D1D; iy++)
               {
                  u_q += u(ix, iy, e) * v1d(qx, ix) * v1d(qy, iy);
                  du_dX_q[0] += u(ix, iy, e) * dv1d_dX(qx, ix) * v1d(qy, iy);
                  du_dX_q[1] += u(ix, iy, e) * v1d(qx, ix) * dv1d_dX(qy, iy);
               }
            }

            // du_dx_q = invJ^T * du_dX_q
            //         = (adjJ^T * du_dX_q) / detJ
            double J_q[2][2] = {{J(qx, qy, 0, 0, e),
                                 J(qx, qy, 0, 1, e)}, // J_q[0][0], J_q[0][1]
                                {J(qx, qy, 1, 0, e),
                                 J(qx, qy, 1, 1, e)}}; // J_q[1][0], J_q[1][1]

            double detJ_q = (J_q[0][0] * J_q[1][1]) - (J_q[0][1] * J_q[1][0]);

            double adjJ[2][2] = {{J_q[1][1], -J_q[0][1]},
                                 {-J_q[1][0], J_q[0][0]}};

            double du_dx_q[2]
               = {(adjJ[0][0] * du_dX_q[0] + adjJ[1][0] * du_dX_q[1]) / detJ_q,
                  (adjJ[0][1] * du_dX_q[0] + adjJ[1][1] * du_dX_q[1]) / detJ_q};

            // auto processed_qf_farg_values = std::apply(
            //    [=](auto... a) {
            //       return std::make_tuple(u_q, du_dx_q, EvaluateFargValue(a)...);
            //    },
            //    qf_farg_values);

            auto [f0, f1] = qf(u_q, du_dx_q);

            double f0_X = f0 * detJ_q;

            // f1_X = invJ * f1 * detJ
            //      = adjJ * f1
            double f1_X[2] = {
               adjJ[0][0] * f1[0] + adjJ[0][1] * f1[1],
               adjJ[1][0] * f1[0] + adjJ[1][1] * f1[1],
            };

            for (int ix = 0; ix < D1D; ix++)
            {
               for (int iy = 0; iy < D1D; iy++)
               {
                  // accumulate v * f0 + dot(dv_dx, f1)
                  y(ix, iy, e) += (f0_X * v1d(qx, ix) * v1d(qy, iy)
                                   + f1_X[0] * dv1d_dX(qx, ix) * v1d(qy, iy)
                                   + f1_X[1] * dv1d_dX(qy, iy) * v1d(qx, ix))
                                  * W(qx, qy);
               }
            }
         }
      }
   }
   // });
}

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::
   ApplyGradient(const Vector &x, const Vector &v, Vector &y) const
{
   ApplyGradient2D(x, v, y);
}

template<typename qfunc_type, typename qfunc_grad_type, typename... qfunc_args_type>
void QFunctionIntegrator<qfunc_type, qfunc_grad_type, qfunc_args_type...>::
   ApplyGradient2D(const Vector &u_in_, const Vector &v_in_, Vector &y_) const
{
   int D1D = dofs1D;
   int Q1D = quad1D;
   int NE = ne;

   auto v1d = Reshape(maps->B.Read(), Q1D, D1D);
   auto dv1d_dX = Reshape(maps->G.Read(), Q1D, D1D);
   // (NQ x SDIM x DIM x NE)
   auto J = Reshape(J_.Read(), Q1D, Q1D, 2, 2, NE);
   auto W = Reshape(W_.Read(), Q1D, Q1D);
   auto u = Reshape(u_in_.Read(), D1D, D1D, NE);
   auto v = Reshape(v_in_.Read(), D1D, D1D, NE);
   auto y = Reshape(y_.ReadWrite(), D1D, D1D, NE);

   for (int e = 0; e < NE; e++)
   {
      // loop over quadrature points
      for (int qy = 0; qy < Q1D; ++qy)
      {
         for (int qx = 0; qx < Q1D; ++qx)
         {
            double u_q = 0.0;
            double du_dX_q[2] = {0.0};
            double v_q = 0.0;
            double dv_dX_q[2] = {0.0};
            for (int ix = 0; ix < D1D; ix++)
            {
               for (int iy = 0; iy < D1D; iy++)
               {
                  u_q += u(ix, iy, e) * v1d(qx, ix) * v1d(qy, iy);

                  du_dX_q[0] += u(ix, iy, e) * dv1d_dX(qx, ix) * v1d(qy, iy);
                  du_dX_q[1] += u(ix, iy, e) * v1d(qx, ix) * dv1d_dX(qy, iy);

                  v_q += v(ix, iy, e) * v1d(qx, ix) * v1d(qy, iy);

                  dv_dX_q[0] += v(ix, iy, e) * dv1d_dX(qx, ix) * v1d(qy, iy);
                  dv_dX_q[1] += v(ix, iy, e) * v1d(qx, ix) * dv1d_dX(qy, iy);
               }
            }

            // du_dx_q = invJ^T * du_dX_q
            //         = (adjJ^T * du_dX_q) / detJ
            double J_q[2][2] = {{J(qx, qy, 0, 0, e),
                                 J(qx, qy, 0, 1, e)}, // J_q[0][0], J_q[0][1]
                                {J(qx, qy, 1, 0, e),
                                 J(qx, qy, 1, 1, e)}}; // J_q[1][0], J_q[1][1]

            double detJ_q = (J_q[0][0] * J_q[1][1]) - (J_q[0][1] * J_q[1][0]);

            double adjJ[2][2] = {{J_q[1][1], -J_q[0][1]},
                                 {-J_q[1][0], J_q[0][0]}};

            double du_dx_q[2]
               = {(adjJ[0][0] * du_dX_q[0] + adjJ[1][0] * du_dX_q[1]) / detJ_q,
                  (adjJ[0][1] * du_dX_q[0] + adjJ[1][1] * du_dX_q[1]) / detJ_q};

            double dv_dx_q[2]
               = {(adjJ[0][0] * dv_dX_q[0] + adjJ[1][0] * dv_dX_q[1]) / detJ_q,
                  (adjJ[0][1] * dv_dX_q[0] + adjJ[1][1] * dv_dX_q[1]) / detJ_q};

            // call Qfunction
            auto [f00, f01, f10, f11] = qf_grad(u_q, du_dx_q);

            double W0 = f00 * v_q + f01[0] * dv_dx_q[0] + f01[1] * dv_dx_q[1];

            double W1[2] = {f10[0] * v_q + f11(0, 0) * dv_dx_q[0]
                               + f11(0, 1) * dv_dx_q[1],
                            f10[1] * v_q + f11(1, 0) * dv_dx_q[0]
                               + +f11(1, 1) * dv_dx_q[1]};

            double W0_X = W0 * detJ_q;

            // W1_X = invJ * W1 * detJ
            //      = adjJ * W1
            double W1_X[2] = {
               adjJ[0][0] * W1[0] + adjJ[0][1] * W1[1],
               adjJ[1][0] * W1[0] + adjJ[1][1] * W1[1],
            };

            for (int ix = 0; ix < D1D; ix++)
            {
               for (int iy = 0; iy < D1D; iy++)
               {
                  // @TODO: proper comment
                  y(ix, iy, e) += (W0_X * v1d(qx, ix) * v1d(qy, iy)
                                   + W1_X[0] * dv1d_dX(qx, ix) * v1d(qy, iy)
                                   + W1_X[1] * dv1d_dX(qy, iy) * v1d(qx, ix))
                                  * W(qx, qy);
               }
            }
         }
      }
   }
}

} // namespace mfem