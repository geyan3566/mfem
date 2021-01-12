#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 0;
   int order = 3;
   bool simplex = true;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinements.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&simplex, "-s", "--simplex", "-no-s", "--no-simplex",
                  "Simplex LOR?");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   Device device(device_config);
   if (mpi.Root()) { device.Print(); }

   Mesh mesh_serial(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; l++)
   {
      mesh_serial.UniformRefinement();
   }

   ParMesh mesh(MPI_COMM_WORLD, mesh_serial);
   mesh_serial.Clear();
   int dim = mesh.Dimension();
   mesh.EnsureNodes();

   H1_FECollection fec_ho(order, dim);
   ParFiniteElementSpace fes_ho(&mesh, &fec_ho);

   
   std::shared_ptr<ParMesh> mesh_lor_tensor(new ParMesh(&mesh, order, BasisType::GaussLobatto));
   std::shared_ptr<ParMesh> mesh_lor;
   if (simplex)
   {
      mesh_lor.reset(new ParMesh);
      mesh_lor->MakeSimplicial(*mesh_lor_tensor);
   }
   else
   {
      mesh_lor = mesh_lor_tensor;
   }

   bool vis = false;
   if (vis)
   {
      mesh_lor->PrintBdrVTU("bdr");
      mesh_lor->PrintVTU("mesh");
   }

   H1_FECollection fec_lor(1, dim);
   ParFiniteElementSpace fes_lor(mesh_lor.get(), &fec_lor);

   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes_ho.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ConstantCoefficient one(1.0);

   ParBilinearForm a_ho(&fes_ho);
   a_ho.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_ho.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_ho.Assemble();
   ParLinearForm b_ho(&fes_ho);
   b_ho.AddDomainIntegrator(new DomainLFIntegrator(one));
   b_ho.Assemble();

   ParBilinearForm a_lor(&fes_lor);
   a_lor.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_lor.Assemble();
   ParLinearForm b_lor(&fes_lor);
   b_lor.AddDomainIntegrator(new DomainLFIntegrator(one));
   b_lor.Assemble();

   ParGridFunction x_ho(&fes_ho), x_lor(&fes_lor);
   x_ho = 0.0;
   x_lor = 0.0;

   Vector X_ho, B_ho, X_lor, B_lor;
   OperatorHandle A_ho, A_lor;
   a_ho.FormLinearSystem(ess_tdof_list, x_ho, b_ho, A_ho, X_ho, B_ho);
   a_lor.FormLinearSystem(ess_tdof_list, x_lor, b_lor, A_lor, X_lor, B_lor);

   HypreBoomerAMG solv_lor;
   solv_lor.SetOperator(*A_lor);
   solv_lor.SetPrintLevel(0);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(200);
   cg.SetPrintLevel(2);
   cg.SetOperator(*A_ho);
   cg.SetPreconditioner(solv_lor);

   if (mpi.Root()) { std::cout << "HO system:  "; }
   X_ho = 0.0;
   cg.Mult(B_ho, X_ho);
   a_ho.RecoverFEMSolution(X_ho, b_ho, x_ho);

   if (mpi.Root()) { std::cout << "LOR system: "; }
   cg.SetOperator(*A_lor);
   X_lor = 0.0;
   cg.Mult(B_lor, X_lor);
   a_lor.RecoverFEMSolution(X_lor, b_lor, x_lor);

   if (vis)
   {
      ParaViewDataCollection dc("LOR", &mesh);
      dc.SetPrefixPath("ParaView");
      dc.SetHighOrderOutput(true);
      dc.SetLevelsOfDetail(order);
      dc.RegisterField("u", &x_ho);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();

      dc.SetMesh(mesh_lor.get());
      dc.DeregisterField("u");
      dc.RegisterField("u", &x_lor);
      dc.SetLevelsOfDetail(1);
      dc.SetCycle(1);
      dc.SetTime(1.0);
      dc.Save();
   }

   return 0;
}