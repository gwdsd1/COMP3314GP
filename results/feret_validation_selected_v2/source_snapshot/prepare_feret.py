"""只使用官方1195对表；官方眼标注 -> 41x27 相似对齐和均衡。"""

import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'.deps'))
import cv2
from gabor import file_hash
from prepare_available_data import alignment_matrix
from run_orl_gray import write_csv

ROOT=BASE/'FERET'
OUTPUT=BASE/'data/processed/feret'
TARGET=np.array([[7.,16.],[20.,16.]])  # 本组约定：画面左/右，(x,y)。
EYE_TARGETS={'wide':[[7.,16.],[20.,16.]],
             'medium':[[8.,15.],[19.,15.]],
             'compact':[[8.5,15.],[18.5,15.]]}


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def load_pairs():
    pairs=read_csv(ROOT/'official_fa_fb_pairs.csv')
    assert len(pairs)==len({r['subject_id'] for r in pairs})==1195
    assert [int(r['official_gallery_order']) for r in pairs]==list(range(1,1196))
    gallery=(ROOT/'metadata/gray_feret_cd2/partitions/by_previously_reported/feret/gallery.names').read_text().split()
    probes=(ROOT/'metadata/gray_feret_cd2/partitions/by_previously_reported/feret/probe_fafb_expression.names').read_text().split()
    assert [Path(r['gallery_file']).stem for r in pairs]==[name for name in gallery if name[:5] in {r['subject_id'] for r in pairs}]
    assert {Path(r['probe_file']).stem for r in pairs}==set(probes)
    return pairs


def variant_images(eye_target,normalization,limit=None):
    """本组候选；验证只读前200人的400张，最终选定后才处理完整集合。"""
    assert eye_target in EYE_TARGETS and normalization in ['equalized','zscore']
    records=read_csv(OUTPUT/'records.csv')
    if limit is not None:records=records[:limit]
    if eye_target=='wide':
        result=np.load(OUTPUT/'aligned.npy')[:len(records)].astype(np.float64)
    else:
        result=[]
        for record in records:
            with Image.open(ROOT/record['file']) as image:raw=np.asarray(image.convert('L'))
            eyes=np.array([[float(record[f'image_{side}_x']),float(record[f'image_{side}_y'])] for side in ['left','right']])
            transform=alignment_matrix(eyes,np.array(EYE_TARGETS[eye_target]))
            warp=cv2.warpAffine(raw,transform,(27,41),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            result.append(np.asarray(ImageOps.equalize(Image.fromarray(warp)),dtype=np.float64))
        result=np.stack(result)
    if normalization=='zscore':
        mean=result.mean(axis=(1,2),keepdims=True);std=result.std(axis=(1,2),keepdims=True)
        if np.any(std<1e-8):raise ValueError('Constant image in zscore normalization')
        result=(result-mean)/std
    assert np.isfinite(result).all()
    return result


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    pairs=load_pairs();candidates={r['file']:r for r in read_csv(ROOT/'fa_fb_candidates.csv')}
    manifest={r['file']:r for r in json.loads((ROOT/'file_manifest.json').read_text())}
    records=[];arrays=[];labels=[];samples=[]
    sample_indices=set(np.linspace(0,1194,16,dtype=int).tolist())
    for person,pair in enumerate(pairs):
        for role,field,ground in [('Fa','gallery_file','gallery_ground_truth'),('Fb','probe_file','probe_ground_truth')]:
            name=pair[field];candidate=candidates[name]
            gnd=ROOT/pair[ground]
            xml=Path(str(gnd).replace('name_value','xml')+'.xml')
            source_paths=[ROOT/name,gnd,xml]
            hashes=[file_hash(p) for p in source_paths]
            assert all(h==manifest[p.relative_to(ROOT).as_posix()]['sha256'] for p,h in zip(source_paths,hashes))
            ET.parse(xml)
            annotation=dict(line.split('=',1) for line in gnd.read_text().splitlines() if '=' in line)
            annotation={k.strip():v.strip() for k,v in annotation.items()}
            assert annotation['id']==candidate['official_subject_id']==pair['subject_id']
            assert annotation['expression']==role.upper()==candidate['official_expression']
            assert pair[ground]==candidate['ground_truth'] and candidate['eyes_in_bounds']=='True'
            eyes=np.array([[float(candidate[f'{side}_eye_x']),float(candidate[f'{side}_eye_y'])] for side in ['left','right']])
            assert np.array_equal(eyes,np.array([list(map(float,annotation[f'{side}_eye_coords'].split())) for side in ['left','right']]))
            eyes=eyes[np.argsort(eyes[:,0])]
            assert np.all((eyes>=0)&(eyes<np.array([256,384]))) and np.linalg.norm(eyes[1]-eyes[0])>10
            with Image.open(ROOT/name) as image:
                assert image.size==(256,384) and image.mode in ['L','P']
                original_mode=image.mode;raw=np.asarray(image.convert('L'))
            transform=alignment_matrix(eyes,TARGET)
            mapped=np.column_stack([eyes,np.ones(2)])@transform.T
            assert np.max(np.abs(mapped-TARGET))<1e-10
            warp=cv2.warpAffine(raw,transform,(27,41),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            valid=cv2.warpAffine(np.ones_like(raw),transform,(27,41),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            aligned=np.asarray(ImageOps.equalize(Image.fromarray(warp)),dtype=np.float32)
            arrays.append(aligned);labels.append(pair['subject_id'])
            records.append({'index':len(records),'official_gallery_order':person+1,'subject_id':pair['subject_id'],
                            'role':role,'file':name,'ground_truth':pair[ground],'xml':xml.relative_to(ROOT).as_posix(),
                            'image_sha256':hashes[0],'ground_truth_sha256':hashes[1],'xml_sha256':hashes[2],
                            'original_mode':original_mode,'image_left_x':eyes[0,0],'image_left_y':eyes[0,1],
                            'image_right_x':eyes[1,0],'image_right_y':eyes[1,1],
                            'source_eye_distance':float(np.linalg.norm(eyes[1]-eyes[0])),
                            'source_support_fraction':float(valid.mean()),
                            'mapped_eye_max_error':float(np.max(np.abs(mapped-TARGET)))})
            if person in sample_indices:samples.append((pair['subject_id'],role,raw,eyes,aligned))
    x=np.stack(arrays);y=np.array(labels)
    assert x.shape==(2390,41,27) and np.isfinite(x).all()
    np.save(OUTPUT/'aligned.npy',x);np.save(OUTPUT/'labels.npy',y)
    write_csv(OUTPUT/'records.csv',records)
    metadata={'source':'official_fa_fb_pairs.csv; official gallery order, filtered to1195 paired identities',
              'shape':list(x.shape),'target_eyes_xy':TARGET.tolist(),'eye_semantics':'sort anatomical left/right by image x',
              'normalization':'geometric similarity only; no per-image zero-mean/unit-variance intensity standardization',
              'warp':'OpenCV bilinear, constant zero boundary, output width27 height41',
              'histogram':'Pillow ImageOps.equalize AFTER alignment; intensity0..255',
              'group_conventions':['target eyes','warp/interpolation/boundary','exact meaning of normalization','filtered official gallery order for first q'],
              'opencv_version':cv2.__version__,'selected_images_and_gnd_and_xml_hashes_checked':7170,
              'source_support_fraction_mean':float(np.mean([r['source_support_fraction'] for r in records])),
              'source_support_fraction_min':min(r['source_support_fraction'] for r in records),
              'max_eye_mapping_error':max(r['mapped_eye_max_error'] for r in records),
              'source_manifest_sha256':file_hash(ROOT/'file_manifest.json'),
              'pair_table_sha256':file_hash(ROOT/'official_fa_fb_pairs.csv'),
              'candidate_table_sha256':file_hash(ROOT/'fa_fb_candidates.csv'),
              'prepared_sha256':{name:file_hash(OUTPUT/name) for name in ['aligned.npy','labels.npy','records.csv']},
              'prepare_code_sha256':file_hash(BASE/'prepare_feret.py')}
    (OUTPUT/'preprocessing.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    canvas=Image.new('RGB',(8*200,4*130),'white');draw=ImageDraw.Draw(canvas)
    for i,(subject,role,raw,eyes,aligned) in enumerate(samples):
        cellx,celly=(i%8)*200,(i//8)*130
        original=Image.fromarray(raw).convert('RGB').resize((64,96));pen=ImageDraw.Draw(original)
        for px,py in eyes:
            px/=4;py/=4;pen.ellipse((px-1,py-1,px+1,py+1),outline='lime')
        canvas.paste(original,(cellx,celly));canvas.paste(Image.fromarray(aligned.astype(np.uint8)).resize((54,82)),(cellx+70,celly+10))
        draw.text((cellx,celly+100),f'{subject} {role}: raw / aligned',fill='black')
    canvas.save(OUTPUT/'alignment_review.png')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':prepare()
