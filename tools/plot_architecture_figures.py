"""Generate thesis architecture/explanation figures (non-data charts)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

FIGS = 'docs/figures'
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# ====== 图2-1: MediaPipe 21点手部骨架 ======
CONNS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
         (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
coords = {
    0:(0.5,0.85), 1:(0.5,0.75), 2:(0.48,0.62), 3:(0.47,0.50), 4:(0.45,0.40),
    5:(0.58,0.72), 6:(0.62,0.58), 7:(0.65,0.44), 8:(0.67,0.32),
    9:(0.55,0.7), 10:(0.56,0.54), 11:(0.58,0.40), 12:(0.60,0.28),
    13:(0.45,0.7), 14:(0.4,0.54), 15:(0.39,0.40), 16:(0.37,0.28),
    17:(0.4,0.72), 18:(0.35,0.58), 19:(0.32,0.44), 20:(0.30,0.32)
}
fig, ax = plt.subplots(figsize=(6, 7))
for a,b in CONNS:
    ax.plot([coords[a][0],coords[b][0]], [coords[a][1],coords[b][1]], 'gray', lw=1.5, alpha=0.6)
colors = plt.cm.tab20(np.linspace(0,1,21))
for i, (name, xy) in enumerate(coords.items()):
    ax.scatter(*xy, s=70, c=[colors[i]], edgecolors='black', linewidth=0.5, zorder=5)
    ax.text(xy[0]+0.015, xy[1]+0.015, str(i), fontsize=6, fontweight='bold')
ax.set_xlim(0.15, 0.8); ax.set_ylim(0.2, 0.95)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('MediaPipe Hand Landmarks (21 Keypoints)', fontweight='bold', pad=10)
fig.savefig(f'{FIGS}/fig_mediapipe_skeleton.png'); plt.close()
print('Fig 2-1 done')

# ====== 图2-2: GCN Message Passing ======
fig, ax = plt.subplots(figsize=(6, 5))
nodes = [(0.5,0.9),(0.3,0.65),(0.7,0.65),(0.2,0.4),(0.5,0.4),(0.8,0.4),(0.3,0.15),(0.7,0.15)]
edges = [(0,1),(0,2),(1,3),(1,4),(2,4),(2,5),(3,6),(4,6),(4,7),(5,7)]
for i,(x,y) in enumerate(nodes):
    ax.scatter(x,y,s=400,c='#2196F3',edgecolors='#1565C0',linewidth=2,zorder=3)
    ax.text(x,y,str(i),ha='center',va='center',color='white',fontweight='bold',fontsize=10)
for a,b in edges:
    x1,y1=nodes[a]; x2,y2=nodes[b]
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#FF9800',lw=2))
    mx,my=(x1+x2)/2,(y1+y2)/2
    ax.text(mx+0.02,my+0.02,'msg',fontsize=7,color='#FF9800',style='italic')
ax.text(0.5,0.98,'Graph Convolution: Message Passing',transform=ax.transAxes,ha='center',fontweight='bold',fontsize=12)
ax.text(0.5,0.93,'Nodes=Joints (21), Edges=Bone Connections (23)',transform=ax.transAxes,ha='center',fontsize=9,color='#666')
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
fig.savefig(f'{FIGS}/fig_gcn_message.png'); plt.close()
print('Fig 2-2 done')

# ====== 图2-5: CE vs CTC ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
for ax, title, color, label, detail in [
    (ax1, 'CE + Temporal Pooling\n(This Work)', '#E3F2FD', 'T frames -> Mean Pool -> 1 Label',
     'Static gesture: whole sequence = one class'),
    (ax2, 'CTC Loss\n(For Seq-to-Seq Tasks)', '#FFF3E0', 'T frames -> Per-frame -> Blank Merge -> L labels',
     'Continuous SLR: needs blank token, alignment')]:
    ax.set_xlim(0,10); ax.set_ylim(0,5); ax.set_title(title, fontweight='bold')
    ax.add_patch(Rectangle((0.5,2.5),8,1.5,fill=True,facecolor=color,edgecolor='#333'))
    ax.text(4.5,3.25,label,ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(4.5,1.8,detail,ha='center',fontsize=8,color='#666'); ax.axis('off')
fig.suptitle('Why CE + Mean Pooling for Static Gesture Recognition',fontweight='bold',y=1.01)
fig.savefig(f'{FIGS}/fig_ce_vs_ctc.png'); plt.close()
print('Fig 2-5 done')

# ====== 图3-3: SpatialGCN Architecture ======
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0,12); ax.set_ylim(0,8)
boxes = [
    (0.5,5.5,2,1.5,'Input\n[21 x 3]','#E3F2FD'),
    (3.5,6.5,2,1,'Angle Encoder\n10 angles -> 64','#C8E6C9'),
    (3.5,4.5,2,1,'GCN Layer 1\n21 -> 128','#BBDEFB'),
    (6,4,2,1,'GCN Layer 2\n128 -> 128','#BBDEFB'),
    (8.5,4,2,1,'GCN Layer 3\n128 -> 256','#BBDEFB'),
    (6,6,2,1,'Concat\n256 + 64','#FFE0B2'),
    (8.5,6,2,1,'FC\n320 -> 256','#FFCC80'),
]
for (bx,by,bw,bh,label,color) in boxes:
    ax.add_patch(FancyBboxPatch((bx,by),bw,bh,boxstyle='round,pad=0.1',facecolor=color,edgecolor='#333',lw=1))
    ax.text(bx+bw/2,by+bh/2,label,ha='center',va='center',fontsize=7,fontweight='bold')
arrows = [(2.5,6.5,3.5,7),(2.5,5.5,3.5,5),(5.5,5,6,4.5),(8,5,8.5,4.5),
          (5.5,7,6,6.5),(5,5,6,5.5),(8,6.5,8.5,6.5),(10.5,6.5,11,6.5)]
for x1,y1,x2,y2 in arrows:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.5))
ax.text(11.3,6.5,'[256]',fontsize=8,fontweight='bold',color='#4CAF50')
ax.set_title('SpatialGCN Architecture (0.15M params)',fontweight='bold'); ax.axis('off')
fig.savefig(f'{FIGS}/fig_spatial_gcn.png'); plt.close()
print('Fig 3-3 done')

# ====== 图3-5: SlowFast TCN ======
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0,14); ax.set_ylim(0,8)
for name, boxes, color, edge in [
    ('Slow Path', [(1.5,5.5,2,1.2,'k=7 d=1'),(4.5,5.5,2,1.2,'k=7 d=2'),(7.5,5.5,2,1.2,'k=7 d=4')],'#BBDEFB','#1565C0'),
    ('Fast Path', [(1.5,2,1.3,1,'k=3 d=1'),(3.2,2,1.3,1,'k=3 d=1'),(4.9,2,1.3,1,'k=3 d=2'),(6.6,2,1.3,1,'k=3 d=2')],'#C8E6C9','#2E7D32')]:
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round',facecolor=color,edgecolor=edge,lw=2))
        ax.text(bx+bw/2,by+bh/2,label,ha='center',va='center',fontsize=7)
ax.add_patch(FancyBboxPatch((10,3),2,3.5,boxstyle='round',facecolor='#FFE0B2',edgecolor='#E65100',lw=2))
ax.text(11,4.75,'Slow-Fast\nFusion +\nOutput Proj',ha='center',va='center',fontsize=8,fontweight='bold')
ax.text(0.3,6.1,'Slow',ha='center',fontsize=9,fontweight='bold',color='#1565C0')
ax.text(0.3,2.5,'Fast',ha='center',fontsize=9,fontweight='bold',color='#2E7D32')
ax.text(0.3,4.3,'[256]',ha='center',fontsize=8)
for yc in [6.1,2.5]:
    px=0.6
    for bx,by,bw,bh,*_ in [(1.5,5.5,2,1.2),(4.5,5.5,2,1.2),(7.5,5.5,2,1.2)] if yc>4 else [(1.5,2,1.3,1),(3.2,2,1.3,1),(4.9,2,1.3,1),(6.6,2,1.3,1)]:
        ax.annotate('',xy=(bx,by+bh/2),xytext=(px,yc),arrowprops=dict(arrowstyle='->',lw=1.5))
        px=bx+bw
    ax.annotate('',xy=(10,4.5),xytext=(px,4.5),arrowprops=dict(arrowstyle='->',lw=1.5))
for fb_x,fb_y,fb_w,fb_h in [(3.2,2,1.3,1),(4.9,2,1.3,1)]:
    cx=fb_x+fb_w/2
    ax.plot([cx,cx],[fb_y+fb_h,5.5],'--',color='#FF9800',lw=1,alpha=0.7)
    ax.text(cx+0.05,3.9,'lat',fontsize=6,color='#FF9800')
ax.text(13,4.75,'[T,14]',fontsize=10,fontweight='bold',color='#4CAF50')
ax.annotate('',xy=(13,4.5),xytext=(12,4.5),arrowprops=dict(arrowstyle='->',lw=2,color='#4CAF50'))
ax.set_title('SlowFast TCN Architecture (0.7M params)',fontweight='bold'); ax.axis('off')
fig.savefig(f'{FIGS}/fig_slowfast_tcn.png'); plt.close()
print('Fig 3-5 done')

# ====== 图3-7: XAI Decision Tree ======
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0,14); ax.set_ylim(0,10)
nodes = [
    (7,9.2,'21 Hand Keypoints','#E3F2FD'),
    (3,7.5,'Finger Extension\n(5 angles > 165deg)','#BBDEFB'),
    (11,7.5,'Palm Orientation\n(thumb direction)','#BBDEFB'),
    (1,5.5,'0 up:\nClosed_Fist','#C8E6C9'),(3,5.5,'1 up:\nGood/One/\nPinky_Up','#C8E6C9'),
    (5,5.5,'2 up:\nTwo/Victory/\nEight/Six','#C8E6C9'),(7,5.5,'3 up:\nSeven/Three','#C8E6C9'),
    (9,5.5,'4 up:\nFour/Nine','#C8E6C9'),(11,5.5,'5 up:\nOpen_Palm','#C8E6C9'),
    (2,3.5,'PIP/MCP\nDual Check\nConf Penalty','#FFE0B2'),
    (8,3.5,'Mode/Content\nClassification\n(disjoint sets)','#FFE0B2'),
    (5,1.8,'15 Gestures + Confidence Score','#FFCC80'),
]
for x,y,label,color in nodes:
    ax.add_patch(FancyBboxPatch((x-1.2,y-0.6),2.4,1.2,boxstyle='round,pad=0.05',facecolor=color,edgecolor='#333',lw=1))
    ax.text(x,y,label,ha='center',va='center',fontsize=6.5,fontweight='bold')
connections = [(7,8.8,3,8.1),(7,8.8,11,8.1),(3,7.1,1,6.1),(3,7.1,3,6.1),(3,7.1,5,6.1),(3,7.1,7,6.1),(3,7.1,9,6.1),(3,7.1,11,6.1),
    (3,5.1,2,4.1),(7,5.1,8,4.1),(2,3.1,5,2.4),(8,3.1,5,2.4)]
for x1,y1,x2,y2 in connections:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1,color='#666'))
ax.set_title('Rule Engine: XAI Decision Tree (0 params, <1ms)',fontweight='bold'); ax.axis('off')
fig.savefig(f'{FIGS}/fig_xai_tree.png'); plt.close()
print('Fig 3-7 done')

print(f'\nAll 6 architecture figures saved to {FIGS}/')
