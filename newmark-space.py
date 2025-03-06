from fenics import *
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True

import sympy2fenics as sf
str2exp = lambda s: sf.sympy2exp(sf.str2sympy(s))


fileP = XDMFFile("outputs/convergence-Viscoelastic-time.xdmf")
fileP.parameters['rewrite_function_mesh']=False # for space convergence true
fileP.parameters["functions_share_mesh"] = True
fileP.parameters["flush_output"] = True

'''
Convergence test for time-discretisation of the wave equation with pure Neumann BCs
Newmark scheme

| rho u_tt - div(grad(u)) = f in Omega
|               grad(u).n = g on Gamma

'''
Tfinal=1
k      = 2
rho    = Constant(1.)
#alpha  = Constant(10.)


# ******* Exact solutions for error analysis ****** #
u_str     = 'exp(-t)*(x*(y-1)*y*(x-1))**2'
dt_u_str  = '(-exp(-t))*(x*(y-1)*y*(x-1))**2'
dtt_u_str = '(exp(-t))*(x*(y-1)*y*(x-1))**2'
v_str = '(1./10*sin(t))*(x*(y-1)*y*(x-1))**2'
# ********* test and trial functions for product space ****** #
nkmax =7

es_0 = []; rs_0 = []; es_div = []; rs_div = []
es = []; rs = []; es_jump = []; rs_jump = []
hh = []

esv_0 = []; rsv_0 = []; esv_div = []; rsv_div = []
esv = []; rsv = []; esv_jump = []; rsv_jump = []

rs.append(0); rs_0.append(0); rs_div.append(0); rs_jump.append(0); 

rsv.append(0); rsv_0.append(0); rsv_div.append(0); rsv_jump.append(0);
# ***** Error analysis ***** #

for nk in range(nkmax):
    nps = pow(2,nk+1)
    mesh = UnitSquareMesh(nps,nps)
    n = FacetNormal(mesh); he = FacetArea(mesh)
    vol = CellVolume (mesh)
    a = 4.0
    sigma = 3.0 * a * k * (k - 1) / 8.0 * he("+") **2 * avg (1 / vol )
    sigma_boundary =3.0 * a * k * (k - 1) * he **2 / vol
    # ********* Finite dimensional spaces ********* #
    P2 = FiniteElement("CG", mesh.ufl_cell(),2)
    Hh = FunctionSpace(mesh,MixedElement([P2,P2]))
   
    print('dofs = ', Hh.dim())
    
    print('h = ',mesh.hmax())
    hh.append(mesh.hmax()) 
    dt = 0.2
    print("....... Refinement level : dt = ", dt)
    
    # ********* instantiation of initial conditions ****** #
    t =0.
    
    u_ex = Expression(str2exp(u_str), t = t, degree=k+4, domain=mesh)
    u_exM = Expression(str2exp(u_str), t = t-dt, degree=k+4, domain=mesh)

    v_ex = Expression(str2exp(v_str), t = t, degree=k+4, domain=mesh)
    v_exM = Expression(str2exp(v_str), t = t-dt, degree=k+4, domain=mesh)


    u_exnm1 = Expression(str2exp(u_str), t = 0, degree=k+4, domain=mesh)
    u_exn = Expression(str2exp(u_str), t = 0, degree=k+4, domain=mesh)
    u_exnp1 = Expression(str2exp(u_str), t = 0, degree=k+4, domain=mesh)
    
    v_exnm1 = Expression(str2exp(v_str), t = 0, degree=k+4, domain=mesh)
    v_exn = Expression(str2exp(v_str), t = 0, degree=k+4, domain=mesh)
    v_exnp1 = Expression(str2exp(v_str), t = 0, degree=k+4, domain=mesh) 
     
    dtt_u_exn = Expression(str2exp(dtt_u_str), t = 0, degree=k+4, domain=mesh)
    dtt_u_exnp1 = Expression(str2exp(dtt_u_str), t = 0, degree=k+4, domain=mesh)
    dtt_u_exnm1 = Expression(str2exp(dtt_u_str), t =0, degree=k+4, domain=mesh)

    u_old = interpolate(u_ex,Hh.sub(0).collapse())
    u_oold = interpolate(u_exM,Hh.sub(0).collapse())
        
    v_old = interpolate(v_ex,Hh.sub(1).collapse())
    v_oold = interpolate(v_exM,Hh.sub(1).collapse())


    # initialise L^infty(0,T;H) norm (max over the time steps)
    E_s0 = 0; E_sdiv = 0; E_sjump=0; E_s = 0
    Ev_s0 = 0; Ev_sdiv = 0; Ev_sjump=0; Ev_s = 0
    
    # ********* Time loop ************* # 
    while (t < Tfinal):

        if t + dt > Tfinal:
            dt = Tfinal - t
            t  = Tfinal
        else:
            t += dt
            
        print("t=%.6f" % t)
        
        sol = Function(Hh)
        dSol = TrialFunction(Hh)
        u,v = split(sol)
        p,q = TestFunctions(Hh)       
 
        def hess(w):
            return grad(grad(w))

        def a_h(w1,w2):
            bilinear = inner(hess(w1),hess(w2))*dx \
                     - inner(avg(dot(hess(w1)*n,n)),jump(grad (w2),n))*dS \
                     - inner(jump(grad(w1),n),avg(dot(hess(w2)*n,n)))*dS \
                     + sigma/he("+")*inner(jump(grad(w1),n),jump(grad(w2),n))*dS\
                     - inner(dot(hess(w1)*n,n),dot(grad(w2),n))*ds\
                     - inner(dot(grad(w1),n),dot(hess(w2)*n,n))*ds \
                     + sigma_boundary/he*inner(dot(grad(w1),n),dot(grad(w2),n))*ds
            return bilinear

        def energynorm(w):
            norm = pow(assemble((((grad(grad(w)))**2)*dx)+(sigma/(he("+")))*((jump(grad(w),n))**2)*dS+((sigma_boundary/(he))*((dot(grad(w),n))**2)*ds)),0.5)
            return norm
        
        def VK(w1, w2):
            return (w1.dx(0).dx(0))*(w2.dx(1).dx(1)) + (w1.dx(1).dx(1))*(w2.dx(0).dx(0)) - 2.0 *(w1.dx(0).dx(1))*(w2.dx(0).dx(1))
        def VKb(w1,w2,w3):
            return w1.dx(0).dx(0)*w2.dx(1)*w3 + w1.dx(1).dx(1)*w2.dx(0)*w3-2*w1.dx(0).dx(1)*w2.dx(0)*w3\
                 - (w1.dx(0).dx(0)*w3).dx(1)*w2 - (w1.dx(1).dx(1)*w3).dx(0)*w2+2*(w1.dx(0).dx(1)*w3).dx(1)*w2      
  
        u_exnp1.t = t;
        u_exn.t = t-dt;
        u_exnm1.t = t-2*dt;
        
        v_exnp1.t = t;
        v_exn.t = t-dt;
        v_exnm1.t = t-2*dt;

        dtt_u_exnp1.t = t;
        dtt_u_exn.t = t-dt;
        dtt_u_exnm1.t = t-2*dt;

        u_ex_mid  = 0.5*(u_exnp1+u_exn)#Expression(str2exp(u_str), t = t-0.5*dt, degree=k+4, domain=mesh)#u_exnp1.(t-0.5*dt)#0.5*(u_exnp1+u_exn)
        dt_uex=1./dt*(u_exnp1-u_exn)
        
        v_exmid  = Expression(str2exp(v_str), t = t-0.5*dt, degree=k+4, domain=mesh)
        dt_vex=(v_exnp1)

        u_exhat =0.25*(u_exnp1+2*u_exn+u_exnm1)
        v_exhat =0.25*(v_exnp1+2*v_exn+v_exnm1)
        u_hat = 0.25*(u+2*u_old+u_oold)
        v_hat = 0.25*(v+2*v_old+v_oold)

        u_bar = 0.5*(u+u_old)

        v_bar=0.5*(v+v_old)
        

        fhat_ex = 0.25*(rho*dtt_u_exnp1 + div(grad(div(grad(u_exnp1)))) \
                        + 2*rho*dtt_u_exn + 2*div(grad(div(grad(u_exn)))) \
                        + rho*dtt_u_exnm1 + div(grad(div(grad(u_exnm1))))\
                        -VK(u_exnp1,v_exnp1) -2*VK(u_exn,v_exn) -VK(u_exnm1,v_exnm1))
       
        g_bar = 0.5*(div(grad(div(grad(v_exnp1))))+div(grad(div(grad(v_exn))))+0.5*VK(u_exnp1,u_exnp1)+0.5*VK(u_exn,u_exn))

        bcu = DirichletBC(Hh.sub(0), u_exnp1, "on_boundary")
        bcv = DirichletBC(Hh.sub(1), v_exnp1, "on_boundary")
        bcH = [bcu,bcv]

        FF = (1/dt**2)*(u-2*u_old+u_oold)*p*dx \
           + 1/rho*a_h(u_hat,p)-0.25*(VK(u,v)*p*dx+2*VK(u_old,v_old)*p*dx+VK(u_oold,v_oold)*p*dx)\
           - 1/rho*fhat_ex * p * dx\
           + a_h(v_bar,q)+0.5*VK(u_bar,u_bar)*q*dx\
           - g_bar*q*dx
  
        Tang = derivative(FF,sol,dSol)
        solve(FF == 0, sol, bcH, J=Tang, \
                solver_parameters={"newton_solver":{"linear_solver":'mumps', "relative_tolerance": 1e-9}})
        u_h,v_h = sol.split()

        u_mid = 0.5*(u_h+u_old)
        dt_u= 1./dt*(u_h-u_old)
                
        v_hmid = 0.5*(v_h+v_old)
       
       # compute L^infty(0,T;H) errors of each contribution at time t_{n+1/2}  
        E_s0 = max(E_s0,pow(assemble((dt_uex-dt_u)**2*dx),0.5))
        E_sdiv= max(E_sdiv,energynorm(u_ex_mid-u_mid))
        
        Ev_s0 = max(Ev_s0,pow(assemble((v_exnp1-v_h)**2*dx),0.5))
        Ev_sdiv= max(Ev_sdiv,energynorm(v_exnp1-v_h))


        assign(u_oold,u_old)
        assign(u_old,u_h)

        assign(v_oold,v_old)
        assign(v_old,v_h)

    
        
    # ********* Storing errors ****** #
    
    es_0.append(E_s0)
    es_div.append(E_sdiv)

    esv_0.append(Ev_s0)
    esv_div.append(Ev_sdiv)
   
      
    if(nk>0):
        rs_0.append(ln(es_0[nk]/es_0[nk-1])/ln(hh[nk]/hh[nk-1]))
        rs_div.append(ln(es_div[nk]/es_div[nk-1])/ln(hh[nk]/hh[nk-1]))
         
        rsv_0.append(ln(esv_0[nk]/esv_0[nk-1])/ln(hh[nk]/hh[nk-1]))
        rsv_div.append(ln(esv_div[nk]/esv_div[nk-1])/ln(hh[nk]/hh[nk-1]))

 # ********* Generating error history ****** #
print('==================================================================================================================')
print('  $h$    &    U-L2    &   Rate   &  U- Energy  &  Rate &    V-L2    &   Rate   &  V- Energy  &  Rate')
print('==================================================================================================================')

print('=====================================================================================')
for nk in range(nkmax):
    print('{:.6f}  & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f} & {:6.2e} & {:.3f}'.format(hh[nk], es_0[nk], rs_0[nk], es_div[nk], rs_div[nk], esv_0[nk], rsv_0[nk], esv_div[nk], rsv_div[nk]))
print('=============================================================================================')
'''
==================================================================================================================
  $h$    &    U-L2    &   Rate   &  U- Energy  &  Rate &    V-L2    &   Rate   &  V- Energy  &  Rate
==================================================================================================================
=====================================================================================
0.707107  & 7.76e-02 & 0.000 & 1.28e-01 & 0.000 & 9.29e-05 & 0.000 & 5.35e-03 & 0.000
0.353553  & 4.17e-02 & 0.897 & 7.85e-02 & 0.700 & 6.23e-05 & 0.577 & 3.79e-03 & 0.499
0.176777  & 1.23e-02 & 1.766 & 3.57e-02 & 1.136 & 2.32e-05 & 1.428 & 2.18e-03 & 0.795
0.088388  & 3.32e-03 & 1.885 & 1.57e-02 & 1.188 & 6.60e-06 & 1.812 & 1.12e-03 & 0.957
0.044194  & 8.43e-04 & 1.977 & 7.04e-03 & 1.154 & 1.70e-06 & 1.959 & 5.62e-04 & 1.001
0.022097  & 2.10e-04 & 2.004 & 3.28e-03 & 1.104 & 4.21e-07 & 2.011 & 2.80e-04 & 1.007
0.011049  & 5.20e-05 & 2.015 & 1.57e-03 & 1.061 & 1.01e-07 & 2.063 & 1.39e-04 & 1.005
0.005524  & 1.18e-05 & 2.142 & 7.73e-04 & 1.022 & 2.53e-08 & 1.993 & 7.00e-05 & 0.992
=============================================================================================
'''
