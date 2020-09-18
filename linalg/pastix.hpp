// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PASTIX
#define MFEM_PASTIX

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PASTIX
#include "operator.hpp"
#include "hypre.hpp"

#include <mpi.h>

#include "pastix.h"

namespace mfem
{

class PastixSparseMatrix : public Operator
{
  public:
    PastixSparseMatrix(const HypreParMatrix & hypParMat);
    ~PastixSparseMatrix();

    void Mult(const Vector &x, Vector &y) const override;
  private:
    spmatrix_t matrix_;
};

class PastixSolver : public Solver
{

};

} // namespace mfem

#endif // MFEM_USE_PASTIX
#endif // MFEM_USE_MPI
#endif // MFEM_PASTIX