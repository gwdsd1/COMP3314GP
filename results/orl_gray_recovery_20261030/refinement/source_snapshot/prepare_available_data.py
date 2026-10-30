"""ORL 双眼几何对齐与均衡；MNIST 每类前500个测试样本。

眼坐标通过独立 Haar 检测器获得，与身份标签和识别分数无关。
检测失败保留 needs_review 状态；后续人工坐标覆盖必须显式保存。
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / ".deps"))
import cv2


TARGET_EYES = np.array([[12.,22.],[34.,22.]])  # 图像左/右，单位为目标像素。


def find_eyes(image, classifiers):
    enlarged = cv2.resize(cv2.equalizeHist(image), None, fx=2, fy=2)
    candidates = []
    for classifier in classifiers:
        for x,y,w,h in classifier.detectMultiScale(enlarged, scaleFactor=1.05, minNeighbors=3,
                                                  minSize=(16,16), maxSize=(66,66)):
            center = np.array([(x+w/2)/2,(y+h/2)/2])
            if 10 < center[0] < 82 and 25 < center[1] < 68:
                if not any(np.linalg.norm(center-other)<3 for other in candidates):
                    candidates.append(center)
    choices = []
    for left in candidates:
        for right in candidates:
            separation = right[0]-left[0]
            if 22 < separation < 56 and abs(right[1]-left[1])<14:
                midpoint=(left+right)/2
                score=2*abs(right[1]-left[1])+0.3*abs(separation-36)+0.3*abs(midpoint[0]-46)+0.1*abs(midpoint[1]-48)
                choices.append((score,left,right))
    if choices:
        score,left,right=min(choices,key=lambda item:item[0])
        return np.stack([left,right]), "auto_haar", float(score)
    # 明确标记待审阅，绝不把此几何候选误称为检测成功。
    smoothed=gaussian_filter(image.astype(float),1.8)
    points=[]
    for x0,x1 in [(17,40),(53,76)]:
        patch=smoothed[36:61,x0:x1]
        y,x=np.unravel_index(np.argmin(patch),patch.shape)
        points.append([x+x0,y+36])
    return np.asarray(points,dtype=float), "needs_review", None


def alignment_matrix(source, target=TARGET_EYES):
    """源眼睛映射到目标眼睛的相似变换：只有平移、旋转、统一缩放。"""
    source, target = np.asarray(source,dtype=float), np.asarray(target,dtype=float)
    vector=source[1]-source[0]; desired=target[1]-target[0]
    if np.linalg.norm(vector)<10: raise ValueError("Invalid eye separation")
    a=np.dot(vector,desired)/np.dot(vector,vector)
    b=(vector[0]*desired[1]-vector[1]*desired[0])/np.dot(vector,vector)
    rotation=np.array([[a,-b],[b,a]])
    return np.column_stack([rotation,target[0]-rotation@source[0]])


def prepare(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    paths=[BASE/f"data/raw/orl/s{p}/{n}.pgm" for p in range(1,41) for n in range(1,11)]
    raw=np.stack([np.asarray(Image.open(p).convert("L")) for p in paths])
    annotations=output/'orl_eye_coordinates.csv'
    canonical=BASE/'data/annotations/orl_eye_coordinates.csv'
    if canonical.exists():annotations.write_bytes(canonical.read_bytes())
    if annotations.exists():
        with annotations.open(encoding='utf-8') as handle: records=list(csv.DictReader(handle))
        assert [r['file'] for r in records]==[str(p.relative_to(BASE/'data/raw/orl')).replace('\\','/') for p in paths]
    else:
        classifiers=[cv2.CascadeClassifier(cv2.data.haarcascades+name)
                     for name in ['haarcascade_eye.xml','haarcascade_eye_tree_eyeglasses.xml']]
        assert all(not c.empty() for c in classifiers)
        records=[]
        for path,image in zip(paths,raw):
            eyes,status,score=find_eyes(image,classifiers)
            records.append({'file':str(path.relative_to(BASE/'data/raw/orl')).replace('\\','/'),
                            'left_x':eyes[0,0],'left_y':eyes[0,1],'right_x':eyes[1,0],'right_y':eyes[1,1],
                            'status':status,'pair_score':score if score is not None else ''})
        with annotations.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    aligned=[]; geometric=[]; errors=[]
    for image,record in zip(raw,records):
        eyes=np.array([[float(record['left_x']),float(record['left_y'])],
                       [float(record['right_x']),float(record['right_y'])]])
        matrix=alignment_matrix(eyes)
        errors.append(float(np.max(np.abs(np.column_stack([eyes,np.ones(2)])@matrix.T-TARGET_EYES))))
        warped=cv2.warpAffine(image,matrix,(46,56),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        geometric.append(warped)
        aligned.append(np.asarray(ImageOps.equalize(Image.fromarray(warped)),dtype=np.float32))
    np.save(output/'orl_aligned.npy',np.stack(aligned))
    labels=np.repeat(np.arange(1,41),10);np.save(output/'orl_labels.npy',labels)
    for sheet in range(10):
        canvas=Image.new('RGB',(10*102,4*134),'white');draw=ImageDraw.Draw(canvas)
        for slot,index in enumerate(range(40*sheet,40*(sheet+1))):
            image=Image.fromarray(raw[index]).convert('RGB');r=records[index]
            pen=ImageDraw.Draw(image)
            color='red' if r['status']=='needs_review' else '#00ff00'
            for k in ['left','right']:
                x,y=float(r[k+'_x']),float(r[k+'_y']);pen.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
            x,y=slot%10*102,slot//10*134;canvas.paste(image,(x,y));draw.text((x,y+113),r['file'][:-4],fill='black')
        canvas.save(output/f'eye_review_{sheet+1:02d}.png')
    canvas=Image.new('RGB',(10*105,4*148),'white');draw=ImageDraw.Draw(canvas)
    for slot,index in enumerate(range(0,400,10)):
        x,y=slot%10*105,slot//10*148
        canvas.paste(Image.fromarray(geometric[index]).resize((92,112)),(x,y))
        draw.text((x,y+115),records[index]['file'][:-4],fill='black')
    canvas.save(output/'aligned_samples.png')

    images=np.frombuffer((BASE/'data/raw/mnist/t10k-images-idx3-ubyte').read_bytes(),dtype=np.uint8,offset=16).reshape(10000,28,28)
    digits=np.frombuffer((BASE/'data/raw/mnist/t10k-labels-idx1-ubyte').read_bytes(),dtype=np.uint8,offset=8)
    indices=np.concatenate([np.flatnonzero(digits==digit)[:500] for digit in range(10)])
    assert len(np.unique(indices))==5000 and np.array_equal(np.bincount(digits[indices]),np.full(10,500))
    np.save(output/'mnist_balanced.npy',images[indices]);np.save(output/'mnist_labels.npy',digits[indices])
    with (output/'mnist_source_indices.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['subset_index','test_index_zero_based','digit'])
        writer.writerows((i,int(j),int(digits[j])) for i,j in enumerate(indices))
    metadata={'orl_target_eyes_xy':TARGET_EYES.tolist(),'eye_methods':{s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})},
              'eye_provenance':'OpenCV Haar candidates, Codex visual review and 16 approximate corrections; not official annotations',
              'max_eye_transform_error_pixels':max(errors),'opencv_version':cv2.__version__,
              'face_warp':'similarity, bilinear, constant-zero padding, equalize after warp',
              'target_eye_convention':'22-pixel spacing and y=22 reduce source extrapolation; geometry choice, not probe optimization',
              'mnist_selection':'first 500 per label in original official 10000-image test order; group convention',
              'eye_annotations_sha256':hashlib.sha256(annotations.read_bytes()).hexdigest()}
    (output/'preprocessing.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    prepare(BASE/'data/processed/available')
