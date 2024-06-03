# Paper 1: FW - Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization
Take aways:
- Convergence of normal FW and novel varient
- Uses duality gap certificates to create better convergence where approximate solutions to the LMO are acceptable  
- Adversarial tacks are as structured norm problem, meaning we can use a special LMO to exploit this structure.

Surrogate Duality Gap:
$$g(x):=\max_{s\in D} \langle x - s, \nabla f(x)\rangle$$
Satisifies definition of dual by convexity and falling under first order approximation of cost: $f(x) + \langle x - s, \nabla f(x)\rangle$.  

Point: FW vs projected gradient comparison. FW subproblems are linear proj grad problems are quadratic.

# Paper 2: FW_variants - On the Global Linear Convergence of Frank-Wolfe Optimization Variants

# Paper 3: FW_survey - Frank–Wolfe and friends: a journey into projection-free first-order optimization methods
- Paper by Rinaldi, et. al.  
- Defines the AA problem in section 3.5  
Maximum allowable $\ell_p$-norm attack  
![alt text](FWV-AAproblem.png)  
Targeted attack: push $x_0$ to target class. Untargeted attack: push $x_0$ away from correct class.

This is where he sayes we should take our problem forumulation from, but this is very general: It does not say which norm we should use, or if it should be targeted. The choice of norm effects what LMO to use, but for targeted/untargeted I suppose there's no reason we can't present results for both.