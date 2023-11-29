import os
os.environ['OMP_NUM_THREADS'] = '1'
import firedrake as df
import numpy as np
import time

from firedrake.petsc import PETSc


def full_quad(order):
    points,weights = np.polynomial.legendre.leggauss(order)
    points = (points+1)/2.
    weights /= 2.
    return points,weights

class VerticalBasis(object):
    def __init__(self,u,H,S_grad,B_grad,p=4,ssa=False):
        self.u = u
        if ssa:
            self.coef = [lambda s: 1.0]
            self.dcoef = [lambda s: 0.0]
        else:
            self.coef = [lambda s:1.0, lambda s:1./p*((p+1)*s**p - 1)]
            self.dcoef = [lambda s:0, lambda s:(p+1)*s**(p-1)]
        
        self.H = H
        self.S_grad = S_grad
        self.B_grad = B_grad

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dz(self,s):
        return self.ds(s)*self.dsdz(s)

    def dx_(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

    def dx(self,s,x):
        return self.dx_(s,x) + self.ds(s)*self.dsdx(s,x)

    def dsdx(self,s,x):
        return 1./self.H*(self.S_grad[x] - s*(self.S_grad[x] - self.B_grad[x]))

    def dsdz(self,x):
        return -1./self.H

class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights

    def integral_term(self,f,s,w):
        return w*f(s)

    def intz(self,f):
        return sum([self.integral_term(f,s,w) 
                    for s,w in zip(self.points,self.weights)])  

class VelocityModel:
    def __init__(
            self,
            mesh,
            solver_type='direct',
            sliding_law='custom',
            vel_scale=1., 
            thk_scale=1.,
            len_scale=1e3,
            beta_scale=1e6,
            time_scale=1,
            pressure_scale=1,
            g=9.81,
            rho_i=917.,
            rho_w=1000.0,
            n=3.0,
            A=1e-16,
            eps_reg=1e-6,
            thklim=1e-3,
            theta=1.0,
            alpha=0,
            p=4,
            c=0.7,
            z_sea=-1000,
            membrane_degree=2,
            shear_degree=3):
            
        self.mesh = mesh
        nhat = df.FacetNormal(mesh)

        E_cg1 = self.E_cg1 = df.FiniteElement('CG',mesh.ufl_cell(),1)
        E_thk = self.E_thk = df.FiniteElement('DG',mesh.ufl_cell(),0)
        velocity_function_space='RT'
        E_vel = self.E_vel = df.FiniteElement(velocity_function_space,
                                                  mesh.ufl_cell(),1)

        E_grd = self.E_grd = df.FiniteElement('RT',mesh.ufl_cell(),1)

        E = self.E = df.MixedElement(E_vel,E_vel)
        
        Q_cg1 = self.Q_cg1 = df.FunctionSpace(mesh,E_cg1)
        Q_vel = self.Q_vel = df.FunctionSpace(mesh,E_vel)
        Q_thk = self.Q_thk = df.FunctionSpace(mesh,E_thk)
        Q_grd = self.Q_grd = df.FunctionSpace(mesh,E_grd)
        V = self.V = df.FunctionSpace(mesh,E)

        self.one = df.Function(Q_thk)
        self.one.assign(1.0)
        self.area = df.assemble(self.one*df.dx)

        theta = self.theta = df.Constant(theta)
        g = self.g = df.Constant(g)
        rho_i = self.rho_i = df.Constant(rho_i)
        rho_w = self.rho_w = df.Constant(rho_w)
        n = self.n = df.Constant(n)
        A = self.A = df.Constant(A)
        eps_reg = self.eps_reg = df.Constant(eps_reg)
        thklim = self.thklim = df.Constant(thklim)
        z_sea = self.z_sea = df.Constant(z_sea)

        vel_scale = self.vel_scale = df.Constant(vel_scale)
        thk_scale = self.thk_scale = df.Constant(thk_scale)
        len_scale = self.len_scale = df.Constant(len_scale)
        beta_scale = self.beta_scale = df.Constant(beta_scale)
        time_scale = self.time_scale = df.Constant(time_scale) 

        eta_star = self.eta_star = df.Constant(A**(-1./n)
                                               * (vel_scale/thk_scale)**((1-n)/n))

        delta = self.delta = df.Constant(thk_scale/len_scale)

        gamma = self.gamma = df.Constant(beta_scale*thk_scale/eta_star)

        omega = self.omega = df.Constant(rho_i*g*thk_scale**3
                                         / (eta_star*len_scale*vel_scale))

        zeta = self.zeta = df.Constant(time_scale*vel_scale/len_scale)


        W = self.W = df.Function(V)
        W_i = self.W_i = df.Function(V)
        Psi = df.TestFunction(V)
        dW = df.TrialFunction(V)

        Ubar,Udef= df.split(W)
        ubar,vbar = Ubar
        udef,vdef = Udef

        Ubar_i,Udef_i = df.split(W_i)
        ubar_i,vbar_i = Ubar_i
        udef_i,vdef_i = Udef_i

        Phibar,Phidef = df.split(Psi)
        phibar_x,phibar_y = Phibar
        phidef_x,phidef_y = Phidef

        S_grad = self.S_grad = df.Function(Q_grd)
        B_grad = self.B_grad = df.Function(Q_grd)
        Chi = df.TestFunction(Q_grd)
        dS = df.TrialFunction(Q_grd)

        self.Ubar0 = df.Function(Q_vel)
        self.Udef0 = df.Function(Q_vel)
        H0 = self.H0 = df.Function(Q_thk,name='H0')
        B = self.B = df.Function(Q_thk,name='B')

        S_lin = self.S_lin = df.Function(Q_thk)  
        B_lin = self.B_lin = df.Function(Q_thk)  
        S_grad_lin = self.S_grad_lin = df.Constant([0.0,0.0])
        B_grad_lin = self.B_grad_lin = df.Constant([0.0,0.0])

        adot = self.adot = df.Function(Q_thk) 
        beta2 = self.beta2 = df.Function(Q_cg1)
        alpha = self.alpha = df.Constant(alpha)

        Bhat = self.Bhat =  df.max_value(B, z_sea - rho_i/rho_w*H0)

        S = self.S = Bhat + H0
        S0 = Bhat + H0
        
        Smid = theta*S + (1-theta)*S0

        self.F_U = F_U = df.Function(Q_vel)
        self.F_H = F_H = df.Function(Q_thk)

        u = VerticalBasis([ubar,udef],H0,S_grad,B_grad,p=p)
        v = VerticalBasis([vbar,vdef],H0,S_grad,B_grad,p=p)
        u_i = VerticalBasis([ubar_i,udef_i],H0,S_grad,B_grad,p=p)
        v_i = VerticalBasis([vbar_i,vdef_i],H0,S_grad,B_grad,p=p)
        phi_x = VerticalBasis([phibar_x,phidef_x],H0,S_grad,B_grad,p=p)
        phi_y = VerticalBasis([phibar_y,phidef_y],H0,S_grad,B_grad,p=p)

        U_b = df.as_vector([u(1),v(1)])
        Phi_b = df.as_vector([phi_x(1),phi_y(1)])

        vi_x = VerticalIntegrator(*full_quad(membrane_degree))
        vi_z = VerticalIntegrator(*full_quad(shear_degree)) 

        def eps_i_II(s):
            return (delta**2*(u_i.dx(s,0))**2 
                        + delta**2*(v_i.dx(s,1))**2 
                        + delta**2*(u_i.dx(s,0))*(v_i.dx(s,1)) 
                        + delta**2*0.25*((u_i.dx(s,1)) + (v_i.dx(s,0)))**2 
                        +0.25*(u_i.dz(s))**2 + 0.25*(v_i.dz(s))**2 
                        + eps_reg)

        def eta(s):
            return 0.5*eps_i_II(s)**((1-n)/(2*n))

        def phi_grad_membrane(s):
            return np.array([[delta*phi_x.dx(s,0), delta*phi_x.dx(s,1)],
                             [delta*phi_y.dx(s,0), delta*phi_y.dx(s,1)]])

        def phi_grad_shear(s):
            return np.array([[phi_x.dz(s)],
                             [phi_y.dz(s)]])

        def phi_outer_membrane(s):
            return np.array([[delta*phi_x(s)*nhat[0],delta*phi_x(s)*nhat[1]],
                             [delta*phi_y(s)*nhat[0],delta*phi_y(s)*nhat[1]]])

        def eps_membrane(s):
            return np.array([[2*delta*u.dx(s,0) + delta*v.dx(s,1), 
                              0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0)],
                             [0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0),
                              delta*u.dx(s,0) + 2*delta*v.dx(s,1)]])

        def eps_shear(s):
            return np.array([[0.5*u.dz(s)],
                            [0.5*v.dz(s)]])

        def membrane_form(s):
            return (2*eta(s)*(eps_membrane(s)
                    * phi_grad_membrane(s)).sum()*H0*df.dx(degree=9))

        def shear_form(s):
            return (2*eta(s)*(eps_shear(s)
                    * phi_grad_shear(s)).sum()*H0*df.dx(degree=9))

        def membrane_boundary_form_nopen(s):
            un = u(s)*nhat[0] + v(s)*nhat[1]
            return alpha*(phi_x(s)*un*nhat[0] + phi_y(s)*un*nhat[1])*df.ds

        def membrane_boundary_form_nat(s):
            return 2*eta(s)*(phi_outer_membrane(s)*eps_membrane(s)).sum()*H0*df.ds

        def membrane_boundary_form_pressure(s):
            return s*omega*H0*(phi_x(s)*nhat[0] + phi_y(s)*nhat[1])*df.ds

      
        membrane_stress = -(vi_x.intz(membrane_form) 
                            + vi_z.intz(shear_form) 
                            + vi_x.intz(membrane_boundary_form_nopen))

        if sliding_law == 'linear':
            basal_stress = -gamma*beta2*df.dot(U_b,Phi_b)*df.dx
        elif sliding_law == 'Budd':
            self.N = (df.min_value(c*H0 + df.Constant(1e-2),
                          H0-rho_w/rho_i*(z_sea - Bhat)) 
                          + df.Constant(1e-4))
            basal_stress = -gamma*beta2*self.N*df.dot(U_b,Phi_b)*df.dx
        elif sliding_law == 'custom':
            self.N = df.Constant(0.15)*rho_i*g*(H0)
            basal_stress = -gamma*beta2*self.N*df.dot(U_b,Phi_b)*df.dx

        driving_stress = (omega*H0*df.dot(S_grad_lin,Phibar)*df.dx 
                          - omega*df.div(Phibar*H0)*(Bhat - S_lin)*df.dx 
                          - omega*df.div(Phibar*H0)*H0*df.dx 
                          + omega*df.jump(Phibar*H0,nhat)*df.avg(Bhat - S_lin)*df.dS 
                          + omega*df.jump(Phibar*H0,nhat)*df.avg(H0)*df.dS 
                          + omega*df.dot(Phibar*H0,nhat)*(Bhat - S_lin)*df.ds 
                          + omega*df.dot(Phibar*H0,nhat)*(H0)*df.ds)

        forcing_stress = df.dot(Phibar,F_U)*df.dx

        R_stress = membrane_stress + basal_stress - driving_stress - forcing_stress


        R_lin = self.R_lin = df.replace(R_stress,{W:dW})

        R_S = (df.dot(Chi,dS)*df.dx 
              - df.dot(Chi,S_grad_lin)*df.dx 
              + df.div(Chi)*(Smid - S_lin)*df.dx 
              - df.dot(Chi,nhat)*(Smid - S_lin)*df.ds)

        R_B = (df.dot(Chi,dS)*df.dx 
              - df.dot(Chi,B_grad_lin)*df.dx 
              + df.div(Chi)*(Bhat - B_lin)*df.dx
              - df.dot(Chi,nhat)*(Bhat - B_lin)*df.ds)

        problem = df.LinearVariationalProblem(df.lhs(R_lin),df.rhs(R_lin),W)

        if solver_type=='direct':
            coupled_parameters = {"ksp_type": "preonly",
                                  "pmat_type":"aij",
                                  "pc_type": "lu",  
                                  "pc_factor_mat_solver_type": "mumps"} 
        else:
            coupled_parameters = {'pc_type': 'bjacobi',
                                  "ksp_rtol":1e-5}

        self.solver = df.LinearVariationalSolver(
            problem,
            solver_parameters=coupled_parameters)
        
        projection_parameters = {'ksp_type':'cg','mat_type':'matfree'}
        S_grad_problem = df.LinearVariationalProblem(df.lhs(R_S),df.rhs(R_S),S_grad)
        self.S_grad_solver = df.LinearVariationalSolver(
            S_grad_problem,
            solver_parameters=projection_parameters)

        B_grad_problem = df.LinearVariationalProblem(df.lhs(R_B),df.rhs(R_B),B_grad)
        self.B_grad_solver = df.LinearVariationalSolver(
            B_grad_problem,
            solver_parameters=projection_parameters)

        self.H_temp = df.Function(self.Q_thk)

    def step(
            self,
            B, 
            H,
            beta2,
            picard_tol=1e-3,
            max_iter=50,
            momentum=0.0,
            error_on_nonconvergence=False,
            convergence_norm='linf',
            forcing=None
        ):
        self.B.assign(B)
        self.H0.assign(H)
        self.beta2.assign(beta2)

        self.W.sub(0).assign(self.Ubar0)
        self.W.sub(1).assign(self.Udef0)

        self.W_i.assign(self.W)

        eps = 1.0
        i = 0
        
        if forcing:
            forcing(self)
        while eps>picard_tol and i<max_iter:
            t_ = time.time()
            self.S_grad_solver.solve()
            self.B_grad_solver.solve()
            self.solver.solve()
            
            if convergence_norm=='linf':
                with self.W_i.dat.vec_ro as w_i:
                    with self.W.dat.vec_ro as w:
                        eps = abs(w_i - w).max()[1]
            else:
                eps = (np.sqrt(
                       df.assemble((self.W_i.sub(2) - self.W.sub(2))**2*df.dx))
                       / self.area)

            PETSc.Sys.Print(i,eps,time.time()-t_)


            self.W_i.assign((1-momentum)*self.W + momentum*self.W_i)
            i+=1

        if i==max_iter and eps>picard_tol:
            converged=False
        else:
            converged=True

        if error_on_nonconvergence and not converged:
            return converged

        self.Ubar0.assign(self.W.sub(0))
        self.Udef0.assign(self.W.sub(1))

        return converged

