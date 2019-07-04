"""
taurus
   N_star    R_hull    m_mean      m_m2      m_m3      m_m4    s_mean  \
0   335.0  4.062128  1.507765  2.724314  4.391856  6.053937  1.698297   

       s_m2      s_m3     s_m4     H_est  sigma_est  
0  1.056308  0.826843  1.34062  0.041692   3.208691  

chai
   N_star    R_hull    m_mean      m_m2      m_m3      m_m4    s_mean  \
0   234.0  5.764974  2.545659  4.023451  6.409064  8.841621  1.662443   

       s_m2      s_m3      s_m4     H_est  sigma_est  
0  1.111883  1.174521  1.666598  0.395697   2.660202  

ic348
   N_star    R_hull   m_mean      m_m2      m_m3      m_m4    s_mean  \
0   350.0  4.933639  2.90091  2.551472  3.467101  4.764035  1.745289   

       s_m2      s_m3      s_m4     H_est  sigma_est  
0  0.976712  0.874018  1.325582  0.790679   1.182075  

lupus3
   N_star    R_hull    m_mean     m_m2      m_m3       m_m4    s_mean  \
0    67.0  5.139166  2.741455  5.38916  9.459509  13.273468  1.428366   

       s_m2      s_m3      s_m4     H_est  sigma_est  
0  1.399918  1.690037  2.191033  0.631318   2.992572  
ophiuchus
   N_star    R_hull    m_mean      m_m2      m_m3      m_m4    s_mean  \
0   198.0  4.439239  2.824132  2.496718  3.564849  4.861686  1.772393   

      s_m2      s_m3      s_m4     H_est  sigma_est  
0  0.92662  0.802028  1.272974  0.690297   1.351324  

ic2391
   N_star   R_hull    m_mean      m_m2      m_m3      m_m4    s_mean  \
0   200.0  3.48549  2.194987  1.347261  1.310091  1.832941  1.798861   

       s_m2      s_m3      s_m4     H_est  sigma_est  
0  0.874127  0.390477  1.079824  0.271176   1.252102  
"""
from clusterfrac.cluster import star_cluster
from maths.points.ra_dec import ra_dec_project
from maths.random.probability_density_function import pdf
from maths.fields.gaussian_random_field import scalar_grf
import numpy as np
from matplotlib import pyplot as plt
from maths.points.fuse_points import fuse_close_companions
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

gridsize=(256,256)
np.random.seed(1)


taurus_dist=140
chai_dist=160
ic348_dist=315
lupus3_dist=170
ophiuchus_dist=130
ic2391_dist=150

cc_dist=0.005





taurus_pos=np.loadtxt("yso_taurus.txt")
taurus_x=ra_dec_project(taurus_pos[:,:3],taurus_pos[:,3:])[:,:2]*taurus_dist
taurus_x=fuse_close_companions(taurus_x,cc_dist)
taurus=star_cluster(taurus_x)

chai_pos=np.loadtxt("yso_chai.txt")
chai_x=ra_dec_project(chai_pos[:,:3],chai_pos[:,3:])[:,:2]*chai_dist
chai_x=fuse_close_companions(chai_x,cc_dist)
chai=star_cluster(chai_x)

ic348_pos=np.loadtxt("yso_ic348.txt")
ic348_x=ra_dec_project(ic348_pos[:,:3],ic348_pos[:,3:])[:,:2]*ic348_dist
ic348_x=fuse_close_companions(ic348_x,cc_dist)
ic348=star_cluster(ic348_x)

lupus3_pos=np.loadtxt("yso_lupus3.txt")
lupus3_x=ra_dec_project(lupus3_pos[:,:3],lupus3_pos[:,3:])[:,:2]*lupus3_dist
lupus3_x=fuse_close_companions(lupus3_x,cc_dist)
lupus3=star_cluster(lupus3_x)

ophiuchus_pos=np.loadtxt("yso_ophiuchus.txt")
ophiuchus_x=ra_dec_project(ophiuchus_pos[:,:3],ophiuchus_pos[:,3:])[:,:2]*ophiuchus_dist
ophiuchus_x=fuse_close_companions(ophiuchus_x,cc_dist)
ophiuchus=star_cluster(ophiuchus_x)

ic2391_pos=np.loadtxt("stars_ic2391.txt")
ic2391_x=ra_dec_project(ic2391_pos[:,:3],ic2391_pos[:,3:])[:,:2]*ic2391_dist
ic2391_x=fuse_close_companions(ic2391_x,cc_dist)
ic2391=star_cluster(ic2391_x)




# Taurus plot

fig_1=plt.figure(figsize=(10,10))
ax=[fig_1.add_subplot(4,4,i+1) for i in range(16)]



ax[4].set_aspect("equal")
ax[4].set_xticks([])
ax[4].set_yticks([])
ax[4].set_xlim(-3,3)
ax[4].set_ylim(-3,3)
ax[4].plot(taurus.r[:,0],taurus.r[:,1],"o",markersize=2.,color="black")

ax[1].set_aspect("equal")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlim(-3,3)
ax[1].set_ylim(-3,3)
ax[1].plot(chai.r[:,0],chai.r[:,1],"o",markersize=2.,color="black")

ax[14].set_aspect("equal")
ax[14].set_xticks([])
ax[14].set_yticks([])
ax[14].set_xlim(-3,3)
ax[14].set_ylim(-3,3)
ax[14].plot(ic348.r[:,0],ic348.r[:,1],"o",markersize=2.,color="black")

ax[11].set_aspect("equal")
ax[11].set_xticks([])
ax[11].set_yticks([])
ax[11].set_xlim(-3,3)
ax[11].set_ylim(-3,3)
ax[11].plot(lupus3.r[:,0],lupus3.r[:,1],"o",markersize=2.,color="black")


A=scalar_grf(gridsize,2.+2.*0.0)
A.normalise(3.2,True)
A.com_shift()
A_pdf=pdf(A.signal.real)
    
fake_cluster=star_cluster(A_pdf.random(335))
    
ax[0].set_aspect("equal")
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlim(-3,3)
ax[0].set_ylim(-3,3)
ax[0].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
    
A=scalar_grf(gridsize,2.+2.*0.4)
A.normalise(2.7,True)
A.com_shift()    
A_pdf=pdf(A.signal.real)
    
fake_cluster=star_cluster(A_pdf.random(234))
    
ax[5].set_aspect("equal")
ax[5].set_xticks([])
ax[5].set_yticks([])
ax[5].set_xlim(-3,3)
ax[5].set_ylim(-3,3)
ax[5].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
A=scalar_grf(gridsize,2.+2.*0.8)
A.normalise(1.2,True)
A.com_shift()

A_pdf=pdf(A.signal.real)
 
fake_cluster=star_cluster(A_pdf.random(350))
    
ax[6].set_aspect("equal")
ax[6].set_xticks([])
ax[6].set_yticks([])
ax[6].set_xlim(-3,3)
ax[6].set_ylim(-3,3)
ax[6].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
A=scalar_grf(gridsize,2.+2.*0.6)
A.normalise(3.0,True)
A.com_shift()
  
A_pdf=pdf(A.signal.real)
   
fake_cluster=star_cluster(A_pdf.random(67))
  
ax[7].set_aspect("equal")
ax[7].set_xticks([])
ax[7].set_yticks([])
ax[7].set_xlim(-3,3)
ax[7].set_ylim(-3,3)
ax[7].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")





A=scalar_grf(gridsize,2.+2.*0.0)
A.normalise(3.2,True)
A.com_shift()
A_pdf=pdf(A.signal.real)
    
fake_cluster=star_cluster(A_pdf.random(335))
    
ax[8].set_aspect("equal")
ax[8].set_xticks([])
ax[8].set_yticks([])
ax[8].set_xlim(-3,3)
ax[8].set_ylim(-3,3)
ax[8].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
    
A=scalar_grf(gridsize,2.+2.*0.4)
A.normalise(2.7,True)
A.com_shift()    
A_pdf=pdf(A.signal.real)
    
fake_cluster=star_cluster(A_pdf.random(234))
    
ax[9].set_aspect("equal")
ax[9].set_xticks([])
ax[9].set_yticks([])
ax[9].set_xlim(-3,3)
ax[9].set_ylim(-3,3)
ax[9].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
A=scalar_grf(gridsize,2.+2.*0.8)
A.normalise(1.2,True)
A.com_shift()

A_pdf=pdf(A.signal.real)
 
fake_cluster=star_cluster(A_pdf.random(350))
    
ax[10].set_aspect("equal")
ax[10].set_xticks([])
ax[10].set_yticks([])
ax[10].set_xlim(-3,3)
ax[10].set_ylim(-3,3)
ax[10].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
A=scalar_grf(gridsize,2.+2.*0.6)
A.normalise(3.0,True)
A.com_shift()
  
A_pdf=pdf(A.signal.real)
   
fake_cluster=star_cluster(A_pdf.random(67))
  
ax[3].set_aspect("equal")
ax[3].set_xticks([])
ax[3].set_yticks([])
ax[3].set_xlim(-3,3)
ax[3].set_ylim(-3,3)
ax[3].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")



A=scalar_grf(gridsize,2.+2.*0.0)
A.normalise(3.2,True)
A.com_shift()
A_pdf=pdf(A.signal.real)
    
fake_cluster=star_cluster(A_pdf.random(335))
    
ax[12].set_aspect("equal")
ax[12].set_xticks([])
ax[12].set_yticks([])
ax[12].set_xlim(-3,3)
ax[12].set_ylim(-3,3)
ax[12].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
    
A=scalar_grf(gridsize,2.+2.*0.4)
A.normalise(2.7,True)
A.com_shift()    
A_pdf=pdf(A.signal.real)
    
fake_cluster=star_cluster(A_pdf.random(234))
    
ax[13].set_aspect("equal")
ax[13].set_xticks([])
ax[13].set_yticks([])
ax[13].set_xlim(-3,3)
ax[13].set_ylim(-3,3)
ax[13].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
A=scalar_grf(gridsize,2.+2.*0.8)
A.normalise(1.2,True)
A.com_shift()

A_pdf=pdf(A.signal.real)
 
fake_cluster=star_cluster(A_pdf.random(350))
    
ax[2].set_aspect("equal")
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_xlim(-3,3)
ax[2].set_ylim(-3,3)
ax[2].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")
    
A=scalar_grf(gridsize,2.+2.*0.6)
A.normalise(3.0,True)
A.com_shift()
  
A_pdf=pdf(A.signal.real)
   
fake_cluster=star_cluster(A_pdf.random(67))
  
ax[15].set_aspect("equal")
ax[15].set_xticks([])
ax[15].set_yticks([])
ax[15].set_xlim(-3,3)
ax[15].set_ylim(-3,3)
ax[15].plot(fake_cluster.r[:,0],fake_cluster.r[:,1],"o",markersize=2.,color="black")


at=AnchoredText(r"$H=0.0, \sigma=3.2$",prop=dict(size=10),frameon=True,loc=2)
at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
ax[0].add_artist(at)

at=AnchoredText(r"$H=0.4, \sigma=2.7$",prop=dict(size=10),frameon=True,loc=2)
at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
ax[1].add_artist(at)

at=AnchoredText(r"$H=0.8, \sigma=1.2$",prop=dict(size=10),frameon=True,loc=2)
at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
ax[2].add_artist(at)

at=AnchoredText(r"$H=0.6, \sigma=3.0$",prop=dict(size=10),frameon=True,loc=2)
at.patch.set_boxstyle("round, pad=0.,rounding_size=0.2")
ax[3].add_artist(at)

fig_1.tight_layout()
fig_1.subplots_adjust(hspace=0,wspace=0)
fig_1.savefig("fakes.png",dpi=150)

