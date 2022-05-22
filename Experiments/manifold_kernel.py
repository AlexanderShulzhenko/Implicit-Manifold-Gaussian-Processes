leg = scipy.special.eval_legendre
class ManifoldKernel(gpflow.kernels.Kernel):

    def __init__(self, nu=3/2, kappa=2, d=3,active_dims=None, dtype=dtype):

        self.dtype = dtype
        self.d = d
        self.nu = gpflow.Parameter(nu, dtype=self.dtype, transform=gpflow.utilities.positive(), name='nu')
        self.kappa = gpflow.Parameter(kappa, dtype=self.dtype, transform=gpflow.utilities.positive(), name='kappa')
        super().__init__(active_dims=active_dims)

        
    def eval_leg(self,n,x):
        base = tf.ones(x.shape, dtype=self.dtype)[None,:,:]
        base1 = x[None,:,:]
        legandre = tf.concat((base,base1), axis = 0)
        for i in range(2,n):
            another_level = (2*i-1)/i * x * legandre[i-1,:,:] - (i-1)/i * legandre[i-2,:,:]
            legandre=tf.concat((legandre, another_level[None,:,:]), axis=0)
        return legandre#[1:,:,:]

    def eval_gegenbauer(self,n,d,x):
        alpha = (d-1)/2
        base = tf.ones(x.shape, dtype=self.dtype)[None,:,:]
        base1 = 2*alpha*x[None,:,:]
        gegen = tf.concat((base,base1), axis = 0)
        for i in range(2,n):
            another_level = (2*x*(i+alpha-1)*gegen[i-1,:,:] - (i+2*alpha-2)*gegen[i-2,:,:])/i
            gegen=tf.concat((gegen, another_level[None,:,:]), axis=0)
        return gegen
    
    def K(self, x, x2=None):
        if x2 is None:
            x2=x
            
        num_feat = 32
        n = np.arange(num_feat) + 1
        #true_eigvals = n * (n+1) #3dim case
        true_eigvals = n * (n+self.d-1) #D-dim case

        dn = (2*n+self.d-1)*gamma(n+self.d-1)/(gamma(n+1)*gamma(self.d))
        cn = dn * gamma((self.d+1)/2) / (2*np.pi**((self.d+1)/2)*ss.eval_gegenbauer(n,(self.d-1)/2,np.ones_like(n)))#self.eval_gegenbauer(n,self.d,np.ones_like(n)))

        psd = tf.pow(2*self.nu/self.kappa**2 + true_eigvals, -self.nu)  # (1, l)
        c_nu = tf.reduce_sum(psd * cn) # (1,)

        cos_geo_dist = tf.matmul(x,tf.transpose(x2))
        eigf = self.eval_gegenbauer(num_feat, self.d, cos_geo_dist)
        k = 1.0/c_nu * tf.reduce_sum(psd[:,None,None] * cn[:,None,None] * eigf, axis=0)

        return k
    
    # not implemented
    def K_diag(self,x):
        pass
