"""Original vector animation and synthesized soundtrack; no external assets."""
from pathlib import Path
import math
import re
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
W, H, S, FPS, SECONDS = 1280, 720, 2, 30, 10
INK = '#254d55'
CREAM = '#fff9e8'
im = None
draw = None
ox = oy = 0


def xy(p):
    return ((p[0] + ox) * S, (p[1] + oy) * S)


def line(points, color=INK, width=3):
    pts = [xy(p) for p in points]
    draw.line(pts, fill=color, width=max(1, round(width * S)), joint='curve')
    r = width * S / 2
    for x, y in (pts[0], pts[-1]):
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)


def ellipse(box, fill, stroke=None, width=2):
    draw.ellipse((*xy(box[:2]), *xy(box[2:])), fill=fill, outline=stroke,
                 width=max(1, round(width*S)))


def path(spec, fill, stroke=None, width=2):
    tokens = re.findall(r'[MLCQZ]|-?\d*\.?\d+', spec)
    points = []
    pos = (0., 0.)
    i = 0
    start = None
    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        n = {'M': 2, 'L': 2, 'C': 6, 'Q': 4, 'Z': 0}[cmd]
        v = list(map(float, tokens[i:i+n]))
        i += n
        if cmd in ('M', 'L'):
            pos = (v[0], v[1])
            points.append(pos)
            if start is None:
                start = pos
        elif cmd == 'Z':
            points.append(start)
        else:
            p0 = np.array(pos)
            p1, p2 = np.array(v[:2]), np.array(v[2:4])
            for f in np.linspace(0, 1, 25)[1:]:
                if cmd == 'C':
                    p3 = np.array(v[4:6])
                    p = (1-f)**3*p0 + 3*(1-f)**2*f*p1 + 3*(1-f)*f*f*p2 + f**3*p3
                else:
                    p = (1-f)**2*p0 + 2*(1-f)*f*p1 + f*f*p2
                points.append(tuple(p))
            pos = points[-1]
    if fill:
        draw.polygon([xy(p) for p in points], fill=fill)
    if stroke:
        line(points, stroke, width)


sky = np.zeros((H*S, W*S, 3), dtype=np.uint8)
top, bottom = np.array([159, 217, 222]), np.array([245, 239, 210])
for y in range(H*S):
    f = min(1., y/(450*S))
    sky[y, :, :] = top*(1-f) + bottom*f
SKY = Image.fromarray(sky)


def cloud(x, y, size=1):
    global ox, oy
    ox, oy = x, y
    # Quiet, rounded silhouettes avoid distracting from the rider.
    ellipse((0, 20*size, 125*size, 59*size), '#f5f6e8')
    ellipse((28*size, 0, 79*size, 55*size), '#f5f6e8')
    ellipse((63*size, 10*size, 111*size, 56*size), '#f5f6e8')
    ox = oy = 0


def scenery(t):
    global ox, oy
    ellipse((994, 65, 1118, 189), '#f7d37c')
    ellipse((1004, 75, 1108, 179), '#ffe4a1')
    for x, y, scale in [(105, 97, 1.25), (410, 51, .8), (875, 146, .65), (1270, 68, 1.1)]:
        cloud((x-10*t+1600) % 1600-140, y, scale)
    path('M 0 327 Q 160 300 290 327 Q 430 306 580 329 Q 800 305 1000 331 L 1280 318 L 1280 480 L 0 480 Z', '#65b7b9')
    path('M 0 350 Q 190 330 410 350 Q 680 330 900 350 Q 1100 329 1280 350 L 1280 474 L 0 474 Z', '#77c8c7')
    for k in range(18):
        y = 358 + (k%5)*18
        x = ((k*173 - t*(18 + k%3*6)) % 1510)-150
        line([(x, y), (x+24+k%4*11, y)], '#b5e0d8', 2)
    # A tiny sailboat travels behind the pelican.
    bx = 235 - t*6
    path(f'M {bx} 382 L {bx+68} 382 Q {bx+52} 397 {bx+13} 393 Z', '#e88d65')
    line([(bx+35, 321), (bx+35, 383)], '#628e8c', 2)
    path(f'M {bx+31} 328 L {bx+31} 377 L {bx-1} 377 Z', '#fff8e6')
    path(f'M {bx+39} 336 L {bx+39} 377 L {bx+63} 377 Z', '#f5d497')
    # Dunes and the beach move more rapidly than the distant water.
    path('M 0 442 Q 230 405 480 439 Q 760 408 960 435 Q 1100 414 1280 429 L 1280 520 L 0 520 Z', '#efd6a4')
    path('M 0 467 Q 200 440 480 464 Q 720 440 980 459 Q 1150 445 1280 458 L 1280 525 L 0 525 Z', '#e9c892')
    for k in range(12):
        x = ((k*137-t*55) % 1520)-120
        y = 459 + k%3*11
        line([(x-5,y+5),(x,y-8),(x+2,y+4)], '#aaa970', 2)
        line([(x+3,y+4),(x+9,y-4)], '#aaa970', 2)
    path('M 0 496 L 1280 496 L 1280 652 L 0 652 Z', '#d8b78c')
    line([(0,501),(1280,501)], '#f6e4bf', 5)
    line([(0,645),(1280,645)], '#f5e4c3', 5)
    for k in range(9):
        x = ((k*200-t*156) % 1800)-240
        line([(x,604),(x+82,604)], '#f1dcbb', 5)
    path('M 0 655 Q 240 636 480 666 Q 700 645 980 665 Q 1150 641 1280 654 L 1280 720 L 0 720 Z', '#81a982')
    for k in range(13):
        x = ((k*127-t*211) % 1600)-150
        y = 683 + k%3*9
        line([(x,y+15),(x+3,y-10),(x+13,y-2)], '#527f6b', 3)
        ellipse((x-2,y-16,x+8,y-6), '#f6dc9b')
        ellipse((x+10,y-7,x+17,y), '#fff3cc')
    for j in range(3):
        x = (144+j*125 + t*13) % 1400-40
        y = 207+j%2*28 + 4*math.sin(t*2+j)
        flap = 6*math.sin(t*5+j)
        path(f'M {x-14} {y-3-flap} Q {x-5} {y-8} {x} {y} Q {x+6} {y-8} {x+15} {y-3-flap}', None, '#60949a', 2)


def wheel(cx, cy, phase):
    r = 78
    ellipse((cx-r,cy-r,cx+r,cy+r), '#eed6b2', INK, 11)
    ellipse((cx-r+7,cy-r+7,cx+r-7,cy+r-7), '#dfc6a2', '#f3e4c7', 4)
    for j in range(12):
        a = phase+j*math.tau/12
        line([(cx,cy),(cx+68*math.cos(a),cy+68*math.sin(a))], '#82918a', 1.3)
    ellipse((cx-9,cy-9,cx+9,cy+9), '#f8e8c4', INK, 3)
    a = phase+.5
    px, py = cx+57*math.cos(a), cy+57*math.sin(a)
    ellipse((px-5,py-3,px+5,py+3), '#f4b566')


def leg(t, far=False):
    phase = t*math.tau*1.2 + (math.pi if far else 0)
    fx, fy = 650+29*math.cos(phase), 524+29*math.sin(phase)
    hip = np.array([633 if far else 652, 408.])
    target = np.array([fx, fy-4])
    v = target-hip
    d = np.linalg.norm(v)
    mid = (hip+target)/2
    normal = np.array([-v[1], v[0]])/d
    knee = mid+normal*math.sqrt(max(0,75**2-d*d/4))
    color = '#bd804e' if far else '#e9a352'
    line([tuple(hip),tuple(knee),tuple(target)], INK, 11)
    line([tuple(hip),tuple(knee),tuple(target)], color, 7)
    # Webbed toes rest on the turning pedals.
    path(f'M {fx-7} {fy-8} Q {fx+3} {fy-10} {fx+7} {fy-3} L {fx+28} {fy+5} L {fx+14} {fy+7} L {fx+4} {fy+4} L {fx-7} {fy+6} Z', color, INK, 2)
    line([(fx-13,fy+10),(fx+22,fy+10)], INK, 5)
    return fx, fy


def bicycle(t):
    global ox, oy
    ox = 10*math.sin(math.tau*t/10)
    oy = 0
    ellipse((421,604,891,624), '#bea77f')
    phase = t*2.
    wheel(505, 535, phase)
    wheel(805, 535, phase)
    leg(t, True)
    # Chain and complete diamond frame.
    path('M 505 525 L 648 506 Q 669 508 668 527 Q 669 544 650 544 L 505 545 Q 492 536 505 525 Z', None, '#6c7770', 3)
    frame = [(505,535),(590,447),(650,524),(505,535)]
    line(frame, INK, 13)
    line(frame, '#2b9997', 8)
    line([(590,447),(754,434),(650,524)], INK, 13)
    line([(590,447),(754,434),(650,524)], '#2b9997', 8)
    line([(805,535),(748,416)], INK, 13)
    line([(805,535),(748,416)], '#2b9997', 8)
    line([(590,447),(578,418)], INK, 8)
    path('M 554 412 Q 580 402 612 414 Q 620 426 600 427 L 560 425 Q 550 422 554 412 Z', '#865b48', INK, 3)
    line([(748,416),(766,391),(800,391)], INK, 7)
    line([(773,389),(803,389)], '#995b48', 9)
    ellipse((757,378,776,390), '#edc163', INK, 2)
    ellipse((636,510,664,538), '#f3dab3', INK, 3)
    fx, fy = 650+29*math.cos(t*math.tau*1.2), 524+29*math.sin(t*math.tau*1.2)
    line([(650,524),(fx,fy)], '#62766f', 6)
    # Front fender has a separate curve and follows the wheel.
    path('M 734 491 C 753 451 817 432 858 472', None, INK, 7)
    path('M 734 491 C 753 451 817 432 858 472', None, '#ef936b', 4)
    path('M 436 489 C 459 450 505 446 539 460', None, INK, 7)
    path('M 436 489 C 459 450 505 446 539 460', None, '#ef936b', 4)
    leg(t)


def pelican(t):
    global oy
    oy = 3*math.sin(t*math.tau*1.2)
    # Tail, body and the rising S-shaped neck form one clear pelican silhouette.
    path('M 535 373 Q 501 375 469 351 L 492 383 L 469 383 Q 516 415 555 408 Z', '#e5eadd', INK, 3)
    path('M 521 345 C 535 310 579 308 627 321 C 660 326 676 320 683 292 C 691 264 700 241 714 219 C 716 188 743 176 765 185 C 788 192 792 217 779 238 C 754 261 745 281 744 315 C 743 357 716 390 679 408 C 658 433 617 444 574 430 C 534 421 508 384 521 345 Z', CREAM, INK, 3)
    path('M 687 284 C 699 310 697 359 674 382 C 657 403 643 415 613 421 C 657 431 695 400 712 372 C 731 344 732 311 737 284 Z', '#e5e8da')
    path('M 720 226 C 708 259 709 281 713 305', None, '#c5d4cb', 2)
    # Distinctive enormous bill and hanging pouch.
    path('M 775 215 L 941 239 Q 929 260 889 278 C 844 300 790 288 773 239 Z', '#efb253', INK, 3)
    path('M 781 238 Q 815 277 868 273 Q 911 265 937 241 L 923 256 Q 876 294 829 284 Q 786 278 781 238 Z', '#f2c76d')
    path('M 774 214 Q 849 217 940 236 Q 952 240 938 245 L 774 235 Z', '#f9ce6e', INK, 3)
    line([(791,232),(926,240)], '#cf8c42', 1.5)
    ellipse((788,220,794,224), '#ac753a')
    # Happy eye and two natural blinks in the loop.
    blink = any(abs(t-b)<.10 for b in (2.2,6.8,9.1))
    if blink:
        path('M 748 212 Q 756 217 764 210', None, INK, 3)
    else:
        ellipse((746,199,765,219), '#fffdf1', INK, 1.5)
        ellipse((753,202,763,215), INK)
        ellipse((755,202,758,206), '#fffdf3')
    path('M 745 193 Q 755 188 764 194', None, INK, 2.2)
    # Small swept crown feathers.
    path('M 720 201 Q 705 184 717 177 Q 722 182 726 187 Q 718 167 730 170 L 741 185', CREAM, INK, 2.5)
    # Breezy coral scarf, animated continuously.
    sway = 7*math.sin(t*5)
    path(f'M 701 278 C 674 {267+sway} 639 {279-sway} 612 {264+sway} L 618 {281+sway} L 604 {287+sway} C 651 {298-sway} 679 {283+sway} 704 291 Z', '#e76e57', INK, 2)
    path('M 702 270 Q 719 281 738 280 L 735 293 Q 715 293 698 283 Z', '#f08c65', INK, 2)
    # A folded wing extends forward to hold the handlebar.
    path('M 548 345 C 578 328 606 337 640 352 C 678 365 709 374 754 380 L 786 382 Q 800 385 799 391 Q 795 396 781 393 Q 795 398 788 403 Q 776 402 765 396 C 703 402 666 389 633 383 C 613 399 584 401 566 388 C 553 377 542 358 548 345 Z', '#ecedde', INK, 3)
    path('M 565 354 Q 591 367 623 366', None, '#bccdc2', 2)
    path('M 562 365 Q 585 380 614 379', None, '#bccdc2', 2)
    path('M 575 383 Q 588 388 601 386', None, '#bccdc2', 2)
    oy = 0


def frame(t):
    global im, draw, ox, oy
    im = SKY.copy()
    draw = ImageDraw.Draw(im)
    ox = oy = 0
    scenery(t)
    bicycle(t)
    pelican(t)
    ox = oy = 0
    return im.resize((W,H), Image.LANCZOS)


def soundtrack():
    sr = 48000
    audio = np.zeros((SECONDS*sr,2), dtype=np.float64)
    rng = np.random.default_rng(73)

    def note(start, midi, duration=.58, gain=.1, pan=0):
        n = int(duration*sr)
        tt = np.arange(n)/sr
        freq = 440*2**((midi-69)/12)
        env = (1-np.exp(-tt*160))*np.exp(-tt*6)
        sig = sum(np.sin(math.tau*freq*h*tt + .015*np.sin(4*tt)) / h**1.7 for h in range(1,6))
        sig = gain*sig*env
        i = int(start*sr)
        m = min(n,len(audio)-i)
        if m>0:
            audio[i:i+m,0] += sig[:m]*(1-pan*.4)
            audio[i:i+m,1] += sig[:m]*(1+pan*.4)

    # Original C-Am-F-G miniature, with plucked harmonics and soft percussion.
    chords = [(48,55,60,64),(45,52,57,60),(41,48,53,57),(43,50,55,59)]
    for beat in range(20):
        chord = chords[(beat//4)%4]
        start = beat*.5
        note(start, chord[0], .7, .08, -.3)
        note(start+.05, chord[2], .5, .055, .3)
        note(start+.25, chord[3], .45, .055, .5)
        n = int(.08*sr)
        ts = np.arange(n)/sr
        shake = rng.normal(0,1,n)*np.exp(-ts*70)*.013
        i = int((start+.25)*sr)
        audio[i:i+n,:] += shake[:,None]
    melody = [76,79,81,79,76,72,74,76,77,76,72,69,74,79,77,74,76,79,84,79]
    for i,midi in enumerate(melody):
        note(i*.5+.1, midi, .55, .045, -.1)
    # Two soft bicycle bell strikes close to the end.
    for start in (7.75,8.02):
        tt = np.arange(int(.9*sr))/sr
        sig = .06*(np.sin(math.tau*1320*tt)+.45*np.sin(math.tau*2944*tt))*np.exp(-tt*5)
        i = int(start*sr)
        audio[i:i+len(tt),:] += sig[:,None]
    fade = np.minimum(np.arange(len(audio))/4800,1)*np.minimum(np.arange(len(audio))[::-1]/12000,1)
    audio *= fade[:,None]
    audio = np.clip(audio,-.95,.95)
    with wave.open(str(OUT/'soundtrack.wav'),'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((audio*32767).astype('<i2').tobytes())


if __name__ == '__main__':
    import sys
    if '--preview' in sys.argv:
        frame(1.2).save(OUT/'preview.png')
    else:
        soundtrack()
        proc = subprocess.Popen([
            'ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
            '-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
            '-i',str(OUT/'soundtrack.wav'),'-c:v','libx264','-preset','medium',
            '-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k',
            '-movflags','+faststart','-shortest',str(OUT/'pelican_bicycle.mp4')
        ],stdin=subprocess.PIPE)
        for i in range(SECONDS*FPS):
            f = frame(i/FPS)
            if i == 36:
                f.save(OUT/'preview.png')
            proc.stdin.write(f.tobytes())
            if i%60==0:
                print(f'Rendered {i}/{SECONDS*FPS}',flush=True)
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError('Video encoding failed')
        print(OUT/'pelican_bicycle.mp4',flush=True)
