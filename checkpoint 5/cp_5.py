from numpy import *
import numpy as np
import math
from PIL import Image


def tonerep_basic(lst):
    max = 0
    for _ in range(len(lst)):
        slst = lst[_]
        for __ in range(len(slst)):
            s = slst[__]
            a = s[0] * 255
            b = s[1] * 255
            c = s[2] * 255
            s = (a, b, c)
            slst[__] = s
        lst[_] = slst
    return lst

def ray_intersection(dx1, dy1, dz1, xip, yip, zip):

    d = np.matrix('0.0,0.0,0.0')
    d[0, 0], d[0, 1], d[0, 2] = dx1, dy1, dz1
    magd = np.linalg.norm(d)
    d[0, 0], d[0, 1], d[0, 2] = d[0, 0] / magd, d[0, 1] / magd, d[0, 2] / magd
    dx1, dy1, dz1 = d[0, 0], d[0, 1], d[0, 2]

    xc, yc, zc = 0.0, 1.6, 5
    r = 3.2

    b = (dx1 * (xip-xc)) + (dy1 * (yip-yc)) + (dz1 * (zip-zc))
    # print b
    b = b * 2
    # print b
    c = ((xip-xc) * (xip-xc)) + ((yip-yc) * (yip-yc)) + ((zip-zc) * (zip-zc)) - (r * r)
    # print (4*c)
    # print ((b * b) - (4*c))
    if ((b * b) - (4 * c) >= 0):
        root = math.sqrt((b * b) - (4 * c))
        w1 = ((-b) + root) / (2)
        w2 = ((-b) - root) / (2)
        #wa = minimum(w1, w2)
        if(w1>=0 and w2<0):
            wa = w1
        elif(w1<0 and w2>=0):
            wa = w2
        elif(w1>=0 and w2>=0):
            wa = minimum(w1, w2)
        else:
            wa = -1
        # print wa
    else:
        wa = -1

    xcs, ycs, zcs = 2.5, -0.5, 6
    rs = 2.9

    b = (dx1 * (xip-xcs)) + (dy1 * (yip-ycs)) + (dz1 * (zip-zcs))
    b = b * 2
    # print b
    c = ((xip - xcs) * (xip - xcs)) + ((yip - ycs) * (yip - ycs)) + ((zip - zcs) * (zip - zcs)) - (rs * rs)
    if ((b * b) - (4 * c) >= 0):
        root = math.sqrt((b * b) - (4 * c))
        w1 = ((-b) + root) / (2)
        w2 = ((-b) - root) / (2)
        #wb = minimum(w1, w2)
        if (w1 >= 0 and w2 < 0):
            wb = w1
        elif (w1 < 0 and w2 >= 0):
            wb = w2
        elif (w1 >= 0 and w2 >= 0):
            wb = minimum(w1, w2)
        else:
            wb = -1
    else:
        wb = -1

    p0 = np.matrix('-5.0,-5.0,0.0')
    p1 = np.matrix('-5.0,-5.0,10.0')
    p2 = np.matrix('5.0,-5.0,0.0')

    e1 = np.matrix('0.0,0.0,0.0')
    e2 = np.matrix('0.0,0.0,0.0')
    t = np.matrix('0.0,0.0,0.0')
    p = np.matrix('0.0,0.0,0.0')
    q = np.matrix('0.0,0.0,0.0')

    e1[0, 0], e1[0, 1], e1[0, 2] = p1[0, 0] - p0[0, 0], p1[0, 1] - p0[0, 1], p1[0, 2] - p0[0, 2]
    e2[0, 0], e2[0, 1], e2[0, 2] = p2[0, 0] - p0[0, 0], p2[0, 1] - p0[0, 1], p2[0, 2] - p0[0, 2]
    t[0, 0], t[0, 1], t[0, 2] = (xip - p0[0, 0]), (yip - p0[0, 1]), (zip - p0[0, 2])
    # print (e1,e2,t)
    p = np.cross(d, e2)
    q = np.cross(t, e1)
    # print(p,q)

    e1t = e1.transpose()
    e2t = e2.transpose()
    tt = t.transpose()
    dt = d.transpose()

    pe1 = np.dot(p, e1t)
    qe2 = np.dot(q, e2t)
    pt = np.dot(p, tt)
    qd = np.dot(q, dt)

    # intersect = True
    if (pe1 == 0):
        intersect = False
    else:
        # intersect = True
        wt = qe2 / pe1
        u = pt / pe1
        v = qd / pe1
        if (wt < 0 or u < 0 or v < 0 or (u + v) > 1):
            intersect = False
        else:
            intersect = True

        xi = ((1 - u - v) * p0[0, 0]) + (u * p1[0, 0]) + (v * p2[0, 0])
        yi = ((1 - u - v) * p0[0, 1]) + (u * p1[0, 1]) + (v * p2[0, 1])
        zi = ((1 - u - v) * p0[0, 2]) + (u * p1[0, 2]) + (v * p2[0, 2])

    p0s = np.matrix('5.0,-5.0,0.0')
    p1s = np.matrix('-5.0,-5.0,10.0')
    p2s = np.matrix('5.0,-5.0,10.0')

    e1s = np.matrix('0.0,0.0,0.0')
    e2s = np.matrix('0.0,0.0,0.0')
    ts = np.matrix('0.0,0.0,0.0')
    ps = np.matrix('0.0,0.0,0.0')
    qs = np.matrix('0.0,0.0,0.0')

    e1s[0, 0], e1s[0, 1], e1s[0, 2] = p1s[0, 0] - p0s[0, 0], p1s[0, 1] - p0s[0, 1], p1s[0, 2] - p0s[0, 2]
    e2s[0, 0], e2s[0, 1], e2s[0, 2] = p2s[0, 0] - p0s[0, 0], p2s[0, 1] - p0s[0, 1], p2s[0, 2] - p0s[0, 2]
    ts[0, 0], ts[0, 1], ts[0, 2] = (xip - p0s[0, 0]), (yip - p0s[0, 1]), (zip - p0s[0, 2])
    # print (e1,e2,t)
    ps = np.cross(d, e2s)
    qs = np.cross(ts, e1s)
    # print(p,q)

    e1ts = e1s.transpose()
    e2ts = e2s.transpose()
    tts = ts.transpose()
    dts = d.transpose()

    pe1s = np.dot(ps, e1ts)
    qe2s = np.dot(qs, e2ts)
    pts = np.dot(ps, tts)
    qds = np.dot(qs, dts)

    # intersect = True
    if (pe1s == 0):
        intersects = False
    else:
        # intersect = True
        wts = qe2s / pe1s
        us = pts / pe1s
        vs = qds / pe1s
        if (wts < 0 or us < 0 or vs < 0 or (us + vs) > 1):
            intersects = False
        else:
            intersects = True

        xis = ((1 - us - vs) * p0s[0, 0]) + (us * p1s[0, 0]) + (vs * p2s[0, 0])
        yis = ((1 - us - vs) * p0s[0, 1]) + (us * p1s[0, 1]) + (vs * p2s[0, 1])
        zis = ((1 - us - vs) * p0s[0, 2]) + (us * p1s[0, 2]) + (vs * p2s[0, 2])

    if(((wb >= 0 and wa >= 0) and wb <= wa) or (wb > 0 and wa < 0)):
        xip1 = xip+(dx1 * wb)
        yip1 = xip+(dy1 * wb)
        zip1 = xip+(dz1 * wb)
        return ('ss', xip1, yip1, zip1, 0, 0)
    elif (wa>=0):
        xip1 = xip+(dx1 * wa)
        yip1 = xip+(dy1 * wa)
        zip1 = xip+(dz1 * wa)
        return('bs', xip1, yip1,zip1,0,0)
    elif(intersect == True):
        return('1t',xi, yi, zi, e1, e2)
    elif(intersects == True):
        return('2t', xis, yis, zis, e1s, e2s)
    #elif(wa <0 and wb<0 and intersect==False and intersects == False):
    else:
        return('nn',0,0,0,0,0)


def reflect(s, n, magn):
    r = np.matrix('0.0,0.0,0.0')

    nt = n.transpose()
    a = np.dot(s, nt)
    a = 2 * a  # / (magn * magn))
    n[0, 0] = n[0, 0] * a
    n[0, 1] = n[0, 1] * a
    n[0, 2] = n[0, 2] * a

    r[0, 0] = s[0, 0] - n[0, 0]
    r[0, 1] = s[0, 1] - n[0, 1]
    r[0, 2] = s[0, 2] - n[0, 2]

    return r[0, 0], r[0, 1], r[0, 2]


def phong(xi, yi, zi, xc, yc, zc, e1, e2, shape, depth = 0, lx=0.0,ly=0.0,lz=0.0):

    #print(xi, yi, zi)
    s = np.matrix('0.0,0.0,0.0')
    s[0, 0], s[0, 1], s[0, 2] = (lx-xi), (ly-yi), (lz-zi)
    mags = np.linalg.norm(s)
    s[0, 0], s[0, 1], s[0, 2] = s[0, 0] / mags, s[0, 1] / mags, s[0, 2] / mags

    n = np.matrix('0.0,0.0,0.0')
    if (shape == 's'):
        n[0, 0], n[0, 1], n[0, 2] = (xi - xc), (yi - yc), (zi - zc)
    else:
        n = np.cross(e1, e2)
    magn = np.linalg.norm(n)
    n[0, 0], n[0, 1], n[0, 2] = n[0, 0] / magn, n[0, 1] / magn, n[0, 2] / magn
    nt = n.transpose()

    a = np.dot(s, nt)

    v = np.matrix('0.0,0.0,0.0')
    v[0, 0], v[0, 1], v[0, 2] = (xi-lx), (yi-ly), (zi-lz)
    magv = np.linalg.norm(v)
    v[0, 0], v[0, 1], v[0, 2] = v[0, 0] / magv, v[0, 1] / magv, v[0, 2] / magv
    vt = v.transpose()

    r = np.matrix('0.0,0.0,0.0')
    r[0, 0], r[0, 1], r[0, 2] = reflect(s, n, magn)
    magr = np.linalg.norm(r)
    r[0, 0], r[0, 1], r[0, 2] = r[0, 0] / magr, r[0, 1] / magr, r[0, 2] / magr

    b = np.dot(r, vt)
    b = b * b

    kd = np.matrix('0.0,0.0,0.0')
    kd[0, 0], kd[0, 1], kd[0, 2] = 0.3, 0, 0

    l = np.matrix('1.0,1.0,1.0')

    ld = np.matrix('0.0,0.0,0.0')
    ld[0, 0], ld[0, 1], ld[0, 2] = l[0, 0] * a, l[0, 1] * a, l[0, 2] * a
    kd[0, 0], kd[0, 1], kd[0, 2] = kd[0, 0] * ld[0, 0], kd[0, 1] * ld[0, 1], kd[0, 2] * ld[0, 2]

    ks = np.matrix('0.4,0.4,0.4')

    ls = np.matrix('0.0,0.0,0.0')
    ls[0, 0], ls[0, 1], ls[0, 2] = l[0, 0] * b, l[0, 1] * b, l[0, 2] * b
    ks[0, 0], ks[0, 1], ks[0, 2] = ks[0, 0] * ls[0, 0], ks[0, 1] * ls[0, 1], ks[0, 2] * ls[0, 2]

    kt = 0.0
    if (shape == 's'):
        if (yc == -0.5):
            kr = 0.95
        else:
            kr = 0.0
    else:
        kr = 0.0

    ill = np.matrix('0.0,0.0,0.0')
    if (depth < 1):
        if (kr > 0):
            #print(xi, yi, zi)
            sh_pe, xip, yip, zip, e1n, e2n = ray_intersection(r[0,0],r[0,1],r[0,2],xi,yi,zi)
            #clr1 = phong(r[0, 0], r[0, 1], r[0, 2], xc, yc, zc, e1, e2, shape, depth + 1)
            if(sh_pe =='bs'):
                clr1 = phong(xip,yip,zip,0.0,1.6,5.0,0,0,'s',depth+1,xi,yi,zi)
            elif(sh_pe =='ss'):
                clr1 = phong(xip, yip, zip, 2.5, -0.5, 6.0, 0, 0, 's', depth + 1, xi, yi, zi)
            elif(sh_pe == '1t'):
                clr1 = phong(xip, yip, zip, 0, 0, 0, e1n, e2n, 't', depth + 1, xi, yi, zi)
            elif(sh_pe == '2t'):
                clr1 = phong(xip, yip, zip, 0, 0, 0, e1n, e2n, 't', depth + 1, xi, yi, zi)
            else:
                clr1 = (0,0,0)
            ill[0, 0], ill[0, 1], ill[0, 2] = clr1[0], clr1[1], clr1[2]
        if (kt > 0):
            pass

    if (kd[0, 0] + ks[0, 0]  < 0):
        kd[0, 0] = 0
        ks[0, 0] = 0
    if (kd[0, 1] + ks[0, 1] < 0):
        kd[0, 1] = 0
        ks[0, 1] = 0
    if (kd[0, 2] + ks[0, 2] < 0):
        kd[0, 2] = 0
        ks[0, 2] = 0


    clr = ((kd[0, 0] + ks[0, 0] + (kr*ill[0,0])), (kd[0, 1] + ks[0, 1] + (kr*ill[0,1])), (kd[0, 2] + ks[0, 2] + (kr*ill[0,2])))
    return clr


def shadow_ray(dx, dy, dz, xc, yc, zc, r, xcs, ycs, zcs, rs):
    d = np.matrix('0.0,0.0,0.0')
    d[0, 0], d[0, 1], d[0, 2] = dx, dy, dz
    # print(dx,dy,dz)
    magd = np.linalg.norm(d)
    d[0, 0], d[0, 1], d[0, 2] = d[0, 0] / magd, d[0, 1] / magd, d[0, 2] / magd
    dx1, dy1, dz1 = d[0, 0], d[0, 1], d[0, 2]

    # b = (dx1 * (-xcs)) + (dy1 * (-ycs)) + (dz1 * (-zcs))
    # print b
    # b = b * 2
    # print b
    # c = (xcs * xcs) + (ycs * ycs) + (zcs * zcs) - (rs * rs)

    # if ((b * b) - (4 * c) < 0):
    clr = phong(dx, dy, dz, xc, yc, zc, 0, 0, 's')
    return clr
    # else:
    #    return (0, 0, 0)


def shadow_ray_triangle(dx, dy, dz, e1, e2):
    d = np.matrix('0.0,0.0,0.0')
    d[0, 0], d[0, 1], d[0, 2] = dx, dy, dz
    # print(dx,dy,dz)
    magd = np.linalg.norm(d)
    d[0, 0], d[0, 1], d[0, 2] = d[0, 0] / magd, d[0, 1] / magd, d[0, 2] / magd
    dx1, dy1, dz1 = d[0, 0], d[0, 1], d[0, 2]
    clr = phong(dx, dy, dz, 0, 0, 0, e1, e2, 't')

    return clr


def tex(t1, t2, t3, u, v, w):

    #u,v,w = int(u),int(v),int(w)
    tx = (u*t1[0,0]) + (v*t2[0,0]) + (w*t3[0,0])
    ty = (u * t1[0, 1]) + (v * t2[0, 1]) + (w * t3[0, 1])
    tx,ty = int(tx),int(ty)

    if((tx%2) ==(ty%2)):
        return (1,0,0)

    return (1,1,0)


def main():
    eyepoint = np.matrix('0.0,0.0,0.0')
    lookat = np.matrix('0.0,0.0,1.0')
    up = np.matrix('0.0,1.0,0.0')
    n = np.matrix('0.0,0.0,0.0')
    u = np.matrix('0.0,0.0,0.0')
    v = np.matrix('0.0,0.0,0.0')
    n[0, 0] = eyepoint[0, 0] - lookat[0, 0]
    n[0, 1] = eyepoint[0, 1] - lookat[0, 1]
    n[0, 2] = eyepoint[0, 2] - lookat[0, 2]
    magn = np.linalg.norm(n)
    n[0, 0], n[0, 1], n[0, 2] = n[0, 0] / magn, n[0, 1] / magn, n[0, 2] / magn
    u = np.cross(up, n)
    magu = np.linalg.norm(u)
    u[0, 0], u[0, 1], u[0, 2] = u[0, 0] / magu, u[0, 1] / magu, u[0, 2] / magu
    v = np.cross(n, u)
    h, w = 10.0, 10.0
    H, W = 200.0, 200.0
    pixel_h = h / (2 * H)
    # print (pixel_h)
    pixel_w = w / (2 * W)
    lst = []

    for i in range(int(H)):
        dy = h / 2 - (((2 * i) + 1) * (pixel_h))
        slst = []
        dz = 3.0
        for j in range(int(W)):
            dx = (-w / 2) + (((2 * j) + 1) * (pixel_w))

            d = np.matrix('0.0,0.0,0.0')
            d[0, 0], d[0, 1], d[0, 2] = dx, dy, dz
            # print(dx,dy,dz)
            magd = np.linalg.norm(d)
            d[0, 0], d[0, 1], d[0, 2] = d[0, 0] / magd, d[0, 1] / magd, d[0, 2] / magd
            dx1, dy1, dz1 = d[0, 0], d[0, 1], d[0, 2]

            xc, yc, zc = 0.0, 1.6, 5.0
            r = 3.2

            b = (dx1 * (-xc)) + (dy1 * (-yc)) + (dz1 * (-zc))
            # print b
            b = b * 2
            # print b
            c = (xc * xc) + (yc * yc) + (zc * zc) - (r * r)
            # print (4*c)
            # print ((b * b) - (4*c))
            if ((b * b) - (4 * c) >= 0):
                root = math.sqrt((b * b) - (4 * c))
                w1 = ((-b) + root) / (2)
                w2 = ((-b) - root) / (2)
                wa = minimum(w1, w2)
                # print wa
            else:
                wa = -1

            xcs, ycs, zcs = 2.5, -0.5, 6
            rs = 2.9

            b = (dx1 * (-xcs)) + (dy1 * (-ycs)) + (dz1 * (-zcs))
            # print b
            b = b * 2
            # print b
            c = (xcs * xcs) + (ycs * ycs) + (zcs * zcs) - (rs * rs)
            # print (4*c)
            # print ((b * b) - (4*c))
            if ((b * b) - (4 * c) >= 0):
                root = math.sqrt((b * b) - (4 * c))
                w1 = ((-b) + root) / (2)
                w2 = ((-b) - root) / (2)
                wb = minimum(w1, w2)
                # print wa
            else:
                wb = -1

            p0 = np.matrix('-5.0,-5.0,0.0')
            p1 = np.matrix('-5.0,-5.0,10.0')
            p2 = np.matrix('5.0,-5.0,0.0')

            e1 = np.matrix('0.0,0.0,0.0')
            e2 = np.matrix('0.0,0.0,0.0')
            t = np.matrix('0.0,0.0,0.0')
            p = np.matrix('0.0,0.0,0.0')
            q = np.matrix('0.0,0.0,0.0')

            e1[0, 0], e1[0, 1], e1[0, 2] = p1[0, 0] - p0[0, 0], p1[0, 1] - p0[0, 1], p1[0, 2] - p0[0, 2]
            e2[0, 0], e2[0, 1], e2[0, 2] = p2[0, 0] - p0[0, 0], p2[0, 1] - p0[0, 1], p2[0, 2] - p0[0, 2]
            t[0, 0], t[0, 1], t[0, 2] = (- p0[0, 0]), (- p0[0, 1]), (- p0[0, 2])
            # print (e1,e2,t)
            p = np.cross(d, e2)
            q = np.cross(t, e1)
            # print(p,q)

            e1t = e1.transpose()
            e2t = e2.transpose()
            tt = t.transpose()
            dt = d.transpose()

            pe1 = np.dot(p, e1t)
            qe2 = np.dot(q, e2t)
            pt = np.dot(p, tt)
            qd = np.dot(q, dt)

            # intersect = True
            if (pe1 == 0):
                intersect = False
            else:
                # intersect = True
                wt = qe2 / pe1
                u = pt / pe1
                v = qd / pe1
                if (wt < 0 or u < 0 or v < 0 or (u + v) > 1):
                    intersect = False
                else:
                    intersect = True

            xi = ((1 - u - v) * p0[0, 0]) + (u * p1[0, 0]) + (v * p2[0, 0])
            yi = ((1 - u - v) * p0[0, 1]) + (u * p1[0, 1]) + (v * p2[0, 1])
            zi = ((1 - u - v) * p0[0, 2]) + (u * p1[0, 2]) + (v * p2[0, 2])

            p0s = np.matrix('5.0,-5.0,0.0')
            p1s = np.matrix('-5.0,-5.0,10.0')
            p2s = np.matrix('5.0,-5.0,10.0')

            e1s = np.matrix('0.0,0.0,0.0')
            e2s = np.matrix('0.0,0.0,0.0')
            ts = np.matrix('0.0,0.0,0.0')
            ps = np.matrix('0.0,0.0,0.0')
            qs = np.matrix('0.0,0.0,0.0')

            e1s[0, 0], e1s[0, 1], e1s[0, 2] = p1s[0, 0] - p0s[0, 0], p1s[0, 1] - p0s[0, 1], p1s[0, 2] - p0s[0, 2]
            e2s[0, 0], e2s[0, 1], e2s[0, 2] = p2s[0, 0] - p0s[0, 0], p2s[0, 1] - p0s[0, 1], p2s[0, 2] - p0s[0, 2]
            ts[0, 0], ts[0, 1], ts[0, 2] = (- p0s[0, 0]), (- p0s[0, 1]), (- p0s[0, 2])
            # print (e1,e2,t)
            ps = np.cross(d, e2s)
            qs = np.cross(ts, e1s)
            # print(p,q)

            e1ts = e1s.transpose()
            e2ts = e2s.transpose()
            tts = ts.transpose()
            dts = d.transpose()

            pe1s = np.dot(ps, e1ts)
            qe2s = np.dot(qs, e2ts)
            pts = np.dot(ps, tts)
            qds = np.dot(qs, dts)

            # intersect = True
            if (pe1s == 0):
                intersects = False
            else:
                # intersect = True
                wts = qe2s / pe1s
                us = pts / pe1s
                vs = qds / pe1s
                if (wts < 0 or us < 0 or vs < 0 or (us + vs) > 1):
                    intersects = False
                else:
                    intersects = True

            xis = ((1 - us - vs) * p0s[0, 0]) + (us * p1s[0, 0]) + (vs * p2s[0, 0])
            yis = ((1 - us - vs) * p0s[0, 1]) + (us * p1s[0, 1]) + (vs * p2s[0, 1])
            zis = ((1 - us - vs) * p0s[0, 2]) + (us * p1s[0, 2]) + (vs * p2s[0, 2])

            t1 = np.matrix('0.0,0.0')
            t2 = np.matrix('0.0,1.0')
            t3 = np.matrix('1.0,0.0')
            t4 = np.matrix('1.0,1.0')

            # print (wa,wb)
            if (wa < 0 and wb < 0 and intersect == False and intersects == False):
                clr = (0, 0, 1)
            elif (wa>=0):
                #xip = (dx1 * wa)
                #yip = (dy1 * wa)
                #zip = (dz1 * wa)
                #clr = shadow_ray(xip, yip, zip, xc, yc, zc, r, xcs, ycs, zcs, rs)
                clr = (0, 0, 1)
            if(wb>=0):
                xip = (dx1 * wb)
                yip = (dy1 * wb)
                zip = (dz1 * wb)
                clr = shadow_ray(xip, yip, zip, xcs, ycs, zcs, rs, xc, yc, zc, r)
            elif (intersect == True):
                #clr = shadow_ray_triangle(xi, yi, zi, e1, e2)
                #clr = tex(t1, t2, t3, u, v, wt)
                clr = (0, 0, 1)
            elif (intersects == True):
            #else:
                #clr = shadow_ray_triangle(xis, yis, zis, e1s, e2s)
                #clr = tex(t3, t2, t4, us, vs, wts)
                clr = (0, 0, 1)
            else:
                clr=(0,0,1)
                #xip = (dx1 * wb)
                #yip = (dy1 * wb)
                #zip = (dz1 * wb)
                #clr = shadow_ray(xip, yip, zip, xcs, ycs, zcs, rs, xc, yc, zc, r)
            slst.append(clr)
        lst.append(slst)
    lst = tonerep_basic(lst)

    arr = np.array(lst, dtype=np.uint8)
    new_image = Image.fromarray(arr)
    new_image.save('cp_51.png')


if __name__ == '__main__':
    main()
