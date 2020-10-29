#include "PML.hpp"

CartesianPML::CartesianPML(Mesh *mesh_, Array2D<double> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void CartesianPML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   // initialize
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = infinity();
      dom_bdr(i, 1) = -infinity();
   }

   for (int i = 0; i < mesh->GetNBE(); i++)
   {
      Array<int> bdr_vertices;
      mesh->GetBdrElementVertices(i, bdr_vertices);
      for (int j = 0; j < bdr_vertices.Size(); j++)
      {
         for (int k = 0; k < dim; k++)
         {
            dom_bdr(k, 0) = min(dom_bdr(k, 0), mesh->GetVertex(bdr_vertices[j])[k]);
            dom_bdr(k, 1) = max(dom_bdr(k, 1), mesh->GetVertex(bdr_vertices[j])[k]);
         }
      }
   }

#ifdef MFEM_USE_MPI
   ParMesh * pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh)
   {
      for (int d=0; d<dim; d++)
      {
         MPI_Allreduce(MPI_IN_PLACE,&dom_bdr(d,0),1,MPI_DOUBLE,MPI_MIN,pmesh->GetComm());
         MPI_Allreduce(MPI_IN_PLACE,&dom_bdr(d,1),1,MPI_DOUBLE,MPI_MAX,pmesh->GetComm());
      }
   }
#endif

   for (int i = 0; i < dim; i++)
   {
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void CartesianPML::SetAttributes(Mesh *mesh_)
{
   int nrelem = mesh_->GetNE();
   elems.SetSize(nrelem);

   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = mesh_->GetElement(i);
      Array<int> vertices;
      // Initialize Attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();
      // Check if any vertex is in the pml
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         double *coords = mesh_->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}

void CartesianPML::StretchFunction(const Vector &x,
                                   vector<complex<double>> &dxs, double omega)
{
   complex<double> zi = complex<double>(0., 1.);

   double n = 2.0;
   // double c = 5.0;
   double c = log(omega);
   double coeff;
   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_dom_bdr(i, 1))
      {
         coeff = n * c / omega / pow(length(i, 1), n);
         dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_dom_bdr(i, 1), n - 1.0));
      }
      if (x(i) <= comp_dom_bdr(i, 0))
      {
         coeff = n * c / omega / pow(length(i, 0), n);
         dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_dom_bdr(i, 0), n - 1.0));
      }
   }
}


ToroidPML::ToroidPML(Mesh *mesh_)
   : mesh(mesh_)
{
   dim = mesh->Dimension();
   zlim.SetSize(2);
   rlim.SetSize(2);
   alim.SetSize(2);
   zpml_thickness.SetSize(2);
   rpml_thickness.SetSize(2);
   apml_thickness.SetSize(2);
   SetBoundaries();
}



void ToroidPML::SetBoundaries()
{
   mesh->EnsureNodes();
   int nrnodes = mesh->GetNodalFESpace()->GetTrueVSize()/dim;
   double zmin = infinity();
   double zmax = -infinity();
   double rmin = infinity();
   double rmax = -infinity();
   double amin = infinity(); // in degrees
   double amax = -infinity(); // in degrees
   for (int i = 0; i<nrnodes; i++)
   {
      Vector coord(dim);
      mesh->GetNode(i,coord);
      for (int d = 0; d<dim; d++)
      {
         if (abs(coord[d])<1e-13) coord[d] = 0.0;
      }
      // Find r and a for this point
      double x = coord[0];
      double y = coord[1];
      double z = coord[2];
      double a = GetAngle(x,y);
      double r = sqrt(x*x + y*y);
      
      zmin = min(zmin,z);
      zmax = max(zmax,z);
      rmin = min(rmin,r);
      rmax = max(rmax,r);
      amin = min(amin,a);
      amax = max(amax,a);
   }

   zlim[0] = zmin;
   zlim[1] = zmax;
   rlim[0] = rmin;
   rlim[1] = rmax;
   alim[0] = amin;
   alim[1] = amax;

#ifdef MFEM_USE_MPI
   ParMesh * pmesh = dynamic_cast<ParMesh *>(mesh);
   if (pmesh)
   {
      MPI_Allreduce(MPI_IN_PLACE,&zlim[0],1,MPI_DOUBLE,MPI_MIN,pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE,&zlim[1],1,MPI_DOUBLE,MPI_MAX,pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE,&rlim[0],1,MPI_DOUBLE,MPI_MIN,pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE,&rlim[1],1,MPI_DOUBLE,MPI_MAX,pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE,&alim[0],1,MPI_DOUBLE,MPI_MIN,pmesh->GetComm());
      MPI_Allreduce(MPI_IN_PLACE,&alim[1],1,MPI_DOUBLE,MPI_MAX,pmesh->GetComm());
   }
#endif
}

void ToroidPML::SetAttributes(Mesh *mesh_)
{
   int nrelem = mesh_->GetNE();
   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      // initialize with 1
      elems[i] = 1;
      Element *el = mesh_->GetElement(i);
      // Initialize attribute
      el->SetAttribute(1);
      Vector center;
      mesh_->GetElementCenter(i,center);
      double x = center[0];
      double y = center[1];
      double a = GetAngle(x,y);

      // check upper and lower bound
      if ( (a <= alim[0]+apml_thickness[0]) ||
           (a >= alim[1]-apml_thickness[1]) )
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   mesh_->SetAttributes();
}


double ToroidPML::GetAngle(const double x, const double y)
{
   // Find r and a for this point
   double arad;
   if (x == 0.0)
   {
      arad = (y > 0.0)? M_PI/2.0 : 3.0 * M_PI/2.0;
   }
   else
   {
      arad = atan(y/x);
      int k = 0;
      if (x<0)
      {
         k = 1;
      }
      else if (y<0)
      {
         k = 2;
      }
      arad += k*M_PI;
   }
   return arad * 180.0/M_PI;
}

void ToroidPML::StretchFunction(const Vector &X,
                                   vector<complex<double>> &dxs, double omega)
{
   complex<double> zi = complex<double>(0., 1.);

   double n = 2.0;
   double c = 5.0;
   // // Stretch in each direction independently
   // for (int i = 0; i < dim; ++i)
   // {
   //    dxs[i] = 1.0;
   //    if (x(i) >= comp_dom_bdr(i, 1))
   //    {
   //       coeff = n * c / omega / pow(length(i, 1), n);
   //       dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_dom_bdr(i, 1), n - 1.0));
   //    }
   //    if (x(i) <= comp_dom_bdr(i, 0))
   //    {
   //       coeff = n * c / omega / pow(length(i, 0), n);
   //       dxs[i] = 1.0 + zi * coeff * abs(pow(x(i) - comp_dom_bdr(i, 0), n - 1.0));
   //    }
   // }
   // Stretch in the azimuthal direction
   double x0 = X[0];
   double y0 = X[1];
   double a0 = GetAngle(x0,y0);
   double r0 = sqrt(x0*x0 + y0*y0);
   dxs[0] = 1.0;
   dxs[1] = 1.0;
   dxs[2] = 1.0;
   double coeff = n * c / omega / pow(1, n);

   // negative direction 
   if (a0 <= alim[0]+apml_thickness[0])
   {
      double a1 = alim[0]*M_PI/180.0;
      double r1 = r0;
      double x1 = r1 * cos(a1);
      double y1 = r1 * sin(a1);
      double xthickness = r1*sin(apml_thickness[0]*M_PI/180.0);
      double ythickness = r1*cos(apml_thickness[0]*M_PI/180.0);
      double coeffx = n * c / omega / pow(xthickness, n);
      double coeffy = n * c / omega / pow(ythickness, n);
      dxs[0] = 1.0 + zi * coeffx * abs(pow(x0 - x1, n - 1.0));
      dxs[1] = 1.0 + zi * coeffy * abs(pow(y0 - y1, n - 1.0));
      cout << "negative " << endl;
   }
   // positive direction 
   if (a0 >= alim[1]-apml_thickness[1])
   {
      double a1 = alim[1]*M_PI/180.0;
      double r1 = r0;
      double x1 = r1 * cos(a1);
      double y1 = r1 * sin(a1);
      double xthickness = r1*sin(apml_thickness[1]*M_PI/180.0);
      double ythickness = r1*cos(apml_thickness[1]*M_PI/180.0);
      xthickness = max(xthickness,1.0);
      ythickness = max(ythickness,1.0);
      double coeffx = n * c / omega / pow(xthickness, n);
      double coeffy = n * c / omega / pow(ythickness, n);
      double Lx = r1 * cos((alim[1]-apml_thickness[1])*M_PI/180.0);
      double Ly = r1 * sin((alim[1]-apml_thickness[1])*M_PI/180.0);
      dxs[0] = 1.0 + zi * coeffx * abs(pow(x0 - Lx, n - 1.0));
      dxs[1] = 1.0 + zi * coeffy * abs(pow(y0 - Ly, n - 1.0));

   }
}

double pml_detJ_Re(const Vector & x, CartesianPML * pml)
{
   int dim = pml->dim;
   double omega = pml->omega;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs, omega);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.real();
}

double pml_detJ_Im(const Vector & x, CartesianPML * pml)
{
   int dim = pml->dim;
   double omega = pml->omega;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs, omega);
   for (int i=0; i<dim; ++i) det *= dxs[i];
   return det.imag();
}

void pml_detJ_JT_J_inv_Re(const Vector & x, CartesianPML * pml , DenseMatrix & M)
{
   int dim = pml->dim;
   double omega = pml->omega;
   std::vector<std::complex<double>> dxs(dim);
   complex<double> det(1.0,0.0);
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i<dim; ++i)
   {
      det *= dxs[i];
   }

   M=0.0;
   for (int i = 0; i<dim; ++i)
   {
      M(i,i) = (det / pow(dxs[i],2)).real();
   }
}

void pml_detJ_JT_J_inv_Im(const Vector & x, CartesianPML * pml , DenseMatrix & M)
{
   int dim = pml->dim;
   double omega = pml->omega;

   std::vector<std::complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i<dim; ++i)
   {
      det *= dxs[i];
   }

   M=0.0;
   for (int i = 0; i<dim; ++i)
   {
      M(i,i) = (det / pow(dxs[i],2)).imag();
   }
}


void detJ_JT_J_inv_Re(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   double omega = pml->omega;
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      M(i, i) = (det / pow(dxs[i], 2)).real();
   }
}

void detJ_JT_J_inv_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   double omega = pml->omega;
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      M(i, i) = (det / pow(dxs[i], 2)).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   double omega = pml->omega;
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   M = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      M(i, i) = abs(det / pow(dxs[i], 2));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   double omega = pml->omega;
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {
      M = (1.0 / det).real();
   }
   else
   {
      M = 0.0;
      for (int i = 0; i < dim; ++i)
      {
         M(i, i) = (pow(dxs[i], 2) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, CartesianPML * pml, DenseMatrix &M)
{
   int dim = pml->dim;
   double omega = pml->omega;
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs, omega);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      M = (1.0 / det).imag();
   }
   else
   {
      M = 0.0;
      for (int i = 0; i < dim; ++i)
      {
         M(i, i) = (pow(dxs[i], 2) / det).imag();
      }
   }
}