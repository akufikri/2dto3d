import cv2, numpy as np, math

img=cv2.imread('sample/raw/image-3.jpg',cv2.IMREAD_GRAYSCALE);img_h,img_w=img.shape
th=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,10)
cl=cv2.morphologyEx(th,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=2)
op=cv2.morphologyEx(cl,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
nl,lbls,st,_=cv2.connectedComponentsWithStats(op,8);mask=np.zeros_like(op)
for i in range(1,nl):
    if st[i,cv2.CC_STAT_AREA]>=50: mask[lbls==i]=255

nl,lbls,st,_=cv2.connectedComponentsWithStats(cl,8);raw2=np.zeros_like(cl)
for i in range(1,nl):
    if st[i,cv2.CC_STAT_AREA]>=100: raw2[lbls==i]=255

skel=cv2.ximgproc.thinning(mask);pruned=skel.copy()
for _ in range(5):
    ep=np.zeros_like(pruned,dtype=np.uint8)
    for y in range(img_h):
        for x in range(img_w):
            if pruned[y,x]==0: continue
            y0,y1=max(0,y-1),min(img_h,y+2);x0,x1=max(0,x-1),min(img_w,x+2)
            if int(np.sum(pruned[y0:y1,x0:x1]>0))-1==1: ep[y,x]=255
    pruned=cv2.subtract(pruned,ep)

lines=cv2.HoughLinesP(pruned,1,np.pi/180,30,30,10)
walls=[]
if lines is not None:
    for ln in lines:
        x1,y1,x2,y2=ln[0];dx,dy=x2-x1,y2-y1
        if math.hypot(dx,dy)<20: continue
        if abs(dy)<abs(dx)*0.3:
            y=(y1+y2)/2;xs,xe=min(x1,x2),max(x1,x2)
            if xe-xs>=20: walls.append({'h':True,'s':(float(xs),float(y)),'e':(float(xe),float(y))})
        elif abs(dx)<abs(dy)*0.3:
            x=(x1+x2)/2;ys,ye=min(y1,y2),max(y1,y2)
            if ye-ys>=20: walls.append({'h':False,'s':(float(x),float(ys)),'e':(float(x),float(ye))})

def merge_ws(ws,ih):
    if not ws: return []
    ws2=sorted(ws,key=lambda w: (round(w['s'][1 if ih else 0]/12), w['s'][0]))
    mg,us=[],[False]*len(ws2)
    for i in range(len(ws2)):
        if us[i]: continue
        cur=dict(ws2[i])
        for j in range(i+1,len(ws2)):
            if us[j]: continue
            o=ws2[j]
            if ih:
                if abs(cur['s'][1]-o['s'][1])>12: continue
                cr=(min(cur['s'][0],cur['e'][0]),max(cur['s'][0],cur['e'][0]))
                or_=(min(o['s'][0],o['e'][0]),max(o['s'][0],o['e'][0]))
            else:
                if abs(cur['s'][0]-o['s'][0])>12: continue
                cr=(min(cur['s'][1],cur['e'][1]),max(cur['s'][1],cur['e'][1]))
                or_=(min(o['s'][1],o['e'][1]),max(o['s'][1],o['e'][1]))
            if not(cr[0]<=or_[1]+10 and or_[0]<=cr[1]+10): continue
            nm,nx=min(cr[0],or_[0]),max(cr[1],or_[1])
            if ih:
                y=round((cur['s'][1]+o['s'][1])/2)
                cur={'h':True,'s':(float(nm),float(y)),'e':(float(nx),float(y))}
            else:
                x=round((cur['s'][0]+o['s'][0])/2)
                cur={'h':False,'s':(float(x),float(nm)),'e':(float(x),float(nx))}
            us[j]=True
        mg.append(cur)
    return mg

hw=[w for w in walls if w['h']];vw=[w for w in walls if not w['h']]
sw=merge_ws(hw,True)+merge_ws(vw,False)

rem=pruned.copy()
for w in sw: cv2.line(rem,(int(w['s'][0]),int(w['s'][1])),(int(w['e'][0]),int(w['e'][1])),0,5)
curved=[]
_,lbls3,st3,_=cv2.connectedComponentsWithStats(rem,8)
for i in range(1,cv2.connectedComponentsWithStats(rem,8)[0]):
    if st3[i,cv2.CC_STAT_AREA]<15: continue
    pts=np.column_stack(np.where(lbls3==i))
    if len(pts)<8: continue
    yp,xp=pts[:,0].astype(np.float64),pts[:,1].astype(np.float64)
    sl=math.hypot(xp[-1]-xp[0],yp[-1]-yp[0])
    if len(pts)/max(sl,1)<1.05: continue
    A=np.column_stack([xp,yp,np.ones(len(xp))]);b=xp**2+yp**2
    try: r,_,_,_=np.linalg.lstsq(A,b,rcond=None)
    except: continue
    cx,cy=r[0]/2,r[1]/2
    if r[2]+cx**2+cy**2<=0: continue
    radius=math.sqrt(r[2]+cx**2+cy**2)
    if radius<8 or radius>500: continue
    rd=np.abs(np.sqrt((xp-cx)**2+(yp-cy)**2)-radius)
    if float(np.sqrt(np.mean(rd**2)))>3: continue
    angs=np.arctan2(yp-cy,xp-cx);sa,ea=float(angs[0]),float(angs[-1])
    if ea<sa: sa,ea=ea,sa
    sp_a=ea-sa
    if sp_a>math.pi: sp_a=2*math.pi-sp_a
    if math.degrees(sp_a)<15: continue
    curved.append({'s':(float(xp[0]),float(yp[0])),'e':(float(xp[-1]),float(yp[-1])),
        'c':(round(cx,1),round(cy,1)),'r':round(radius,1),'sa':sa,'ea':ea,'sp':round(math.degrees(sp_a),1)})

oc,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
rc,_=cv2.findContours(raw2,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
ocnt=max(oc,key=cv2.contourArea);rpts=max(rc,key=cv2.contourArea).reshape(-1,2)

peri=cv2.arcLength(ocnt,True)
aps=[tuple(p[0]) for p in cv2.approxPolyDP(ocnt,0.008*peri,True)]

fillets=[];seen=set()
for k in range(len(aps)):
    p1,p2,p3=aps[(k-1)%len(aps)],aps[k],aps[(k+1)%len(aps)]
    dx1,dy1=p2[0]-p1[0],p2[1]-p1[1];dx2,dy2=p3[0]-p2[0],p3[1]-p2[1]
    l1,l2=math.hypot(dx1,dy1),math.hypot(dx2,dy2)
    if l1<15 or l2<15: continue
    ang=math.acos(max(-1.0,min(1.0,(dx1*dx2+dy1*dy2)/(l1*l2))))
    if math.degrees(ang)<45 or math.degrees(ang)>135: continue
    d=np.sum((rpts.astype(float)-np.array(p2,float))**2,axis=1);ni=int(np.argmin(d))
    for h in (10,12,15):
        s_=(ni-h)%len(rpts);e_=(ni+h+1)%len(rpts)
        seg=np.vstack([rpts[s_:],rpts[:e_]]) if s_>e_ else rpts[s_:e_]
        if len(seg)<5: continue
        x=seg[:,0].astype(np.float64);y=seg[:,1].astype(np.float64)
        A=np.column_stack([x,y,np.ones(len(x))]);b=x**2+y**2
        try: r,_,_,_=np.linalg.lstsq(A,b,rcond=None)
        except: continue
        cx,cy=r[0]/2,r[1]/2
        if r[2]+cx**2+cy**2<=0: continue
        radius=math.sqrt(r[2]+cx**2+cy**2)
        if cx<-50 or cx>img_w+50 or cy<-50 or cy>img_h+50: continue
        if math.hypot(cx-p2[0],cy-p2[1])>30: continue
        if radius<6 or radius>50: continue
        rd=np.abs(np.sqrt((x-cx)**2+(y-cy)**2)-radius)
        if float(np.sqrt(np.mean(rd**2)))>3: continue
        angs2=np.arctan2(y-cy,x-cx);sa2,ea2=float(angs2[0]),float(angs2[-1])
        if ea2<sa2: sa2,ea2=ea2,sa2
        sp2=ea2-sa2
        if sp2>math.pi or math.degrees(sp2)<30: continue
        key=(round(cx,1),round(cy,1))
        if key in seen: continue
        fillets.append({'s':(round(cx+radius*math.cos(sa2),1),round(cy+radius*math.sin(sa2),1)),
            'e':(round(cx+radius*math.cos(ea2),1),round(cy+radius*math.sin(ea2),1)),
            'c':key,'r':round(radius,1),'sa':sa2,'ea':ea2,'sp':round(math.degrees(sp2),1)})
        seen.add(key);break

print(f"=== Image-3 Final ===")
print(f"Straight: {len(sw)}")
for i,w in enumerate(sw):
    l=math.hypot(w['e'][0]-w['s'][0],w['e'][1]-w['s'][1])
    print(f"  W{i:2d}: ({w['s'][0]:4.0f},{w['s'][1]:4.0f})->({w['e'][0]:4.0f},{w['e'][1]:4.0f}) {'H' if w['h'] else 'V'} {l:3.0f}px")
print(f"Curved:  {len(curved)}")
for i,c in enumerate(curved):
    print(f"  C{i:2d}: cen=({c['c'][0]:5.1f},{c['c'][1]:5.1f}) r={c['r']:4.1f} sp={c['sp']:4.1f}deg")
print(f"Fillets: {len(fillets)}")
for i,f in enumerate(fillets):
    print(f"  F{i:2d}: cen=({f['c'][0]:5.1f},{f['c'][1]:5.1f}) r={f['r']:4.1f} sp={f['sp']:4.1f}deg")

vis=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
for w in sw:
    cv2.line(vis,(int(w['s'][0]),int(w['s'][1])),(int(w['e'][0]),int(w['e'][1])),(0,255,0),3)
    cv2.circle(vis,(int(w['s'][0]),int(w['s'][1])),4,(255,0,0),-1)
    cv2.circle(vis,(int(w['e'][0]),int(w['e'][1])),4,(0,0,255),-1)
for c in curved:
    cp=(int(c['c'][0]),int(c['c'][1]))
    cv2.ellipse(vis,cp,(int(c['r']),int(c['r'])),0,int(math.degrees(c['sa'])),int(math.degrees(c['ea'])),(255,0,255),3)
    cv2.circle(vis,cp,4,(0,255,255),-1)
for f in fillets:
    cp=(int(f['c'][0]),int(f['c'][1]))
    cv2.ellipse(vis,cp,(int(f['r']),int(f['r'])),0,int(math.degrees(f['sa'])),int(math.degrees(f['ea'])),(0,255,255),3)
    cv2.circle(vis,cp,4,(255,0,255),-1)
cv2.imwrite("debug_img3_result.png",vis)
print("\nSaved debug_img3_result.png")
