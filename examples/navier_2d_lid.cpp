// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// 3D flow over a cylinder benchmark example

#include "mfem.hpp"
#include "../fluid/navier_solver.hpp"
#include <fstream>

using namespace mfem;
using namespace navier;
using namespace std;

struct s_NavierContext
{
   int order = 4;
   double kin_vis = 0.001;
   double t_final = 8.0;
   double dt = 1e-3;
} ctx;

void vel(const Vector &x, double t, Vector &u)
{
   double xi = x(0);
   double yi = x(1);

   double U = 1.0;

   if ( abs(yi - 1.0) <= 1e-8)
   {
      u(0) = U;
   }
   else
   {
      u(0) = 0.0;
   }
   u(1) = 0.0;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi(argc, argv);

	const char *mesh_file = "../data/cavity_origin.mesh";
   int serial_refinements = 3;
   OptionsParser args(argc, argv);
	args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&serial_refinements, "-r", "--refine",
                  "Level of refinement");
    args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   Mesh *mesh = new Mesh(mesh_file);

   for (int i = 0; i < serial_refinements; ++i)
   {
      mesh->UniformRefinement();
   }

   if (mpi.Root())
   {
      std::cout << "Number of elements: " << mesh->GetNE() << std::endl;
   }

   auto *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Create the flow solver.
   NavierSolver flowsolver(pmesh, ctx.order, ctx.kin_vis);
   flowsolver.EnablePA(true);

   // Set the initial condition.
   ParGridFunction *u_ic = flowsolver.GetCurrentVelocity();
   VectorFunctionCoefficient u_excoeff(pmesh->Dimension(), vel);
   u_ic->ProjectCoefficient(u_excoeff);

   // Add Dirichlet boundary conditions to velocity space restricted to
   // selected attributes on the mesh.
   Array<int> attr(pmesh->bdr_attributes.Max());
   // Inlet is attribute 1.
   // attr[1] = 1;
   // Walls is attribute 3.
   attr[1] = 1;
   flowsolver.AddVelDirichletBC(vel, attr);

   ofstream save_mesh("lid_driven_initial.vtk");
   save_mesh.precision(14);
   pmesh->PrintVTK(save_mesh, 0);
   u_ic->SaveVTK(save_mesh, "initial", 0);
   save_mesh.close();

   double t = 0.0;
   double dt = ctx.dt;
   double t_final = ctx.t_final;
   bool last_step = false;

   flowsolver.Setup(dt);

   ParGridFunction *u_gf = flowsolver.GetCurrentVelocity();
   ParGridFunction *p_gf = flowsolver.GetCurrentPressure();

   ParaViewDataCollection pvdc("lid", pmesh);
   pvdc.SetDataFormat(VTKFormat::BINARY32);
   pvdc.SetHighOrderOutput(true);
   pvdc.SetLevelsOfDetail(ctx.order);
   pvdc.SetCycle(0);
   pvdc.SetTime(t);
   pvdc.RegisterField("velocity", u_gf);
   pvdc.RegisterField("pressure", p_gf);
   pvdc.Save();

   for (int step = 0; !last_step; ++step)
   {
      if (t + dt >= t_final - dt / 2)
      {
         last_step = true;
      }

      flowsolver.Step(t, dt, step);

      if (step % 10 == 0)
      {
         pvdc.SetCycle(step);
         pvdc.SetTime(t);
         pvdc.Save();
      }

      if (mpi.Root())
      {
         printf("%11s %11s\n", "Time", "dt");
         printf("%.5E %.5E\n", t, dt);
         fflush(stdout);
      }
   }

   flowsolver.PrintTimingData();

   delete pmesh;

   return 0;
}
