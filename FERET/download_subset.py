"""NIST remote tar index / selected-file download. Credentials are read from stdin.

No archive executable is run. Range responses and archive version are checked.
Index checkpoints allow interruption without downloading the full archive.
"""
import argparse
import bz2
import concurrent.futures
import csv
import hashlib
import http.client
import io
import json
import re
import sys
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath

BASE=Path(__file__).resolve().parent
URL='https://nigos.nist.gov/colorferet/colorferet.tar'
SIZE=8579819520
START=7365130752
ETAG='"1ff65a800-4cee191a02e40"'
CHUNK=8*1024*1024
WINDOW=128*1024
LOCK=threading.Lock()
BYTES=0
AUTH=None
CONNECTIONS=threading.local()


def fetch(start,end):
    global BYTES
    end=min(end,SIZE-1)
    for attempt in range(5):
        try:
            if not getattr(CONNECTIONS,'connection',None):
                CONNECTIONS.connection=http.client.HTTPSConnection('nigos.nist.gov',timeout=35)
            connection=CONNECTIONS.connection
            connection.request('GET','/colorferet/colorferet.tar',headers={'Authorization':AUTH,
                'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'curl/8.0'})
            response=connection.getresponse()
            if response.status!=206 or response.getheader('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                raise ValueError('Server did not honor requested byte range')
            if response.getheader('ETag')!=ETAG:raise ValueError('Archive version changed')
            data=response.read()
            if len(data)!=end-start+1:raise ValueError('Incomplete range response')
            with LOCK:BYTES+=len(data)
            return data
        except Exception as error:
            if getattr(CONNECTIONS,'connection',None):CONNECTIONS.connection.close()
            CONNECTIONS.connection=None
            if attempt==4:raise RuntimeError(f'Range failed at {start}: {type(error).__name__}') from None
            time.sleep(min(2**attempt,8))


def header(data,offset):
    if data[offset+257:offset+262]!=b'ustar':return None
    try:
        item=tarfile.TarInfo.frombuf(data[offset:offset+512],'utf-8','replace')
        if item is None:return None
        return {'name':item.name,'size':item.size,'type':item.type.decode('ascii','replace')}
    except (tarfile.HeaderError,ValueError):return None


def scan(begin,end,known_start=None):
    cursor=known_start if known_start is not None else begin
    entries=[];buffer=b'';origin=-1;last_name=''
    while cursor<end:
        if not origin<=cursor<origin+len(buffer)-511:
            amount=512 if not last_name or '/data/images/' in last_name else 65536
            origin=cursor;buffer=fetch(cursor,min(cursor+amount-1,SIZE-1))
        index=cursor-origin
        item=header(buffer,index)
        if item is None:
            # Two zero blocks are the tar end marker; do not scan archive padding.
            if buffer[index:index+512]==b'\0'*512 and known_start is not None:
                following=fetch(cursor+512,cursor+1023)
                if following==b'\0'*512:break
            found=None
            for i in range(index+512,len(buffer)-511,512):
                candidate=header(buffer,i)
                if candidate:
                    found=(i,candidate);break
            if found is None:
                cursor=origin+(len(buffer)//512)*512
                continue
            index,item=found;cursor=origin+index
        if cursor>=end:break
        item['offset']=cursor;entries.append(item)
        last_name=item['name']
        cursor+=512+((item['size']+511)//512)*512
    return entries


def initial_scan(begin,end):
    data=fetch(begin,min(begin+WINDOW-1,SIZE-1))
    for i in range(0,len(data)-511,512):
        item=header(data,i)
        if item:return scan(begin,end,begin+i)
    return None


def index_archive():
    checkpoint=BASE/'index_checkpoint.jsonl'
    ranges=[(b,min(b+CHUNK,SIZE)) for b in range(START,SIZE,CHUNK)]
    print(json.dumps({'status':'index_start','chunks':len(ranges),'window_bytes':WINDOW}),flush=True)
    completed={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                record=json.loads(line);completed[record['begin']]=record
    def save(begin,end,entries):
        record={'begin':begin,'end':end,'entries':entries}
        with checkpoint.open('a',encoding='utf-8') as handle:handle.write(json.dumps(record)+'\n')
        completed[begin]=record
        print(json.dumps({'indexed_chunks':len(completed),'total_chunks':len(ranges),
                          'received_MB_this_process':round(BYTES/1e6,1)}),flush=True)
    missing=[(b,e) for b,e in ranges if b not in completed];deferred=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures={executor.submit(initial_scan,b,e):(b,e) for b,e in missing}
        for future in concurrent.futures.as_completed(futures):
            b,e=futures[future];entries=future.result()
            if entries is None:deferred.append((b,e))
            else:save(b,e,entries)
    # Empty probes usually lie inside large unused software / test files.
    # A previously verified tar header proves the payload span contains no headers.
    spans=sorted((x['offset']+512,x['offset']+512+((x['size']+511)//512)*512)
                 for r in completed.values() for x in r['entries'] if x['size']>=WINDOW)
    for b,e in sorted(deferred):
        covering=[last for first,last in spans if first<=b<last]
        cursor=max(covering) if covering else b
        entries=[] if cursor>=e else scan(b,e,cursor)
        save(b,e,entries)
        spans.extend((x['offset']+512,x['offset']+512+((x['size']+511)//512)*512)
                     for x in entries if x['size']>=WINDOW)
    entries=sorted((x for r in completed.values() for x in r['entries']),key=lambda x:x['offset'])
    assert len({x['offset'] for x in entries})==len(entries)
    manifest={'url':URL,'archive_bytes':SIZE,'etag':ETAG,'scanned_start':START,
              'mode':'HTTP byte ranges; no complete tar retained','entries':entries}
    (BASE/'archive_index.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'status':'index_complete','entries':len(entries),'received_MB_this_process':BYTES/1e6}),flush=True)


def payload(entry):
    return fetch(entry['offset']+512,entry['offset']+512+entry['size']-1) if entry['size'] else b''


def safe_save(relative,data):
    name=PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe archive path')
    path=BASE.joinpath(*name.parts)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError(f'Existing file differs: {relative}')
    else:
        temporary=path.with_name(path.name+'.part');temporary.write_bytes(data);temporary.replace(path)
    return {'file':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def download_candidates():
    from PIL import Image
    entries=json.loads((BASE/'archive_index.json').read_text())['entries']
    lookup={e['name']:e for e in entries}
    pattern=re.compile(r'^(\d{5})(fa|fb|ba|bj)\d{3}[a-z]?_\d{6}\.tif\.bz2$')
    images=[e for e in entries if e['type']=='0' and '/data/images/' in e['name'] and pattern.match(PurePosixPath(e['name']).name)]
    assert len({PurePosixPath(e['name']).name for e in images})==len(images)
    selected={e['name']:e for e in images}
    for image in images:
        for directory,suffix in [('name_value','.gnd'),('xml','.gnd.xml')]:
            name=image['name'].replace('/images/',f'/ground_truths/{directory}/').removesuffix('.tif.bz2')+suffix
            if name not in lookup:raise ValueError('Missing original ground truth: '+name)
            selected[name]=lookup[name]
    for name,item in lookup.items():
        if '/gray_feret_cd' in name and item['type']=='0' and (
                PurePosixPath(name).name in ['feret_master.html','feret_master.pdf','FERET_Database_Release_Agreement_feb2001.pdf']
                or '/partitions/by_facial_expression/' in name
                or '/partitions/by_previously_reported/feret/' in name and ('gallery' in name or 'probe_fafb' in name or PurePosixPath(name).name=='README')):
            selected[name]=item
    checkpoint=BASE/'download_checkpoint.jsonl';finished={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                record=json.loads(line);p=BASE/record['file']
                if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']:
                    finished[record['archive_member']]=record
    pending=[e for e in sorted(selected.values(),key=lambda e:e['offset']) if e['name'] not in finished]
    groups=[]
    for e in pending:
        begin=e['offset']+512;end=begin+e['size']
        if groups and begin-groups[-1]['end']<=8192 and end-groups[-1]['begin']<=512*1024:
            groups[-1]['items'].append(e);groups[-1]['end']=end
        else:groups.append({'begin':begin,'end':end,'items':[e]})
    print(json.dumps({'status':'candidate_download_start','image_candidates':len(images),
                      'selected_files':len(selected),'resumed_files':len(finished),'range_groups':len(groups)}),flush=True)
    def one(group):
        data=fetch(group['begin'],group['end']-1)
        for entry in group['items']:
            position=entry['offset']+512-group['begin'];content=data[position:position+entry['size']]
            compressed_sha=hashlib.sha256(content).hexdigest()
            extra={}
            if entry['name'].endswith('.tif.bz2'):
                content=bz2.decompress(content)  # Includes bzip2 stream CRC validation.
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    grayscale_palette=False
                    if image.mode=='P':
                        palette=image.getpalette('RGB')
                        grayscale_palette=bool(palette) and all(palette[i]==palette[i+1]==palette[i+2]
                                                              for i in range(0,len(palette),3))
                    if image.format!='TIFF' or not (image.mode=='L' or grayscale_palette):
                        raise ValueError(f"Unexpected image format: {entry['name']} {image.format} {image.mode}")
                    pixel_range=image.convert('L').getextrema() if grayscale_palette else image.getextrema()
                    extra={'kind':'image','width':image.width,'height':image.height,'mode':image.mode,
                           'grayscale_palette':grayscale_palette,
                           'pixel_min':pixel_range[0],'pixel_max':pixel_range[1],
                           'bz2_crc_verified':True}
                relative='images/'+PurePosixPath(entry['name']).name.removesuffix('.bz2')
            else:
                relative='metadata/'+entry['name'].removeprefix('colorferet/dvd2/')
                extra={'kind':'metadata'}
            record={**safe_save(relative,content),**extra,'archive_member':entry['name'],
                    'archive_data_offset':entry['offset']+512,'archive_compressed_bytes':entry['size'],
                    'archive_payload_sha256':compressed_sha}
            with LOCK:
                with checkpoint.open('a',encoding='utf-8') as handle:handle.write(json.dumps(record)+'\n')
                finished[entry['name']]=record
                if len(finished)%200==0:print(json.dumps({'downloaded_files':len(finished),'total':len(selected),
                                                        'received_MB':round(BYTES/1e6,1)}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for _ in executor.map(one,groups):pass
    assert set(finished)==set(selected)
    records=sorted(finished.values(),key=lambda r:r['archive_member'])
    (BASE/'file_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    rows=[]
    for item in images:
        name=PurePosixPath(item['name']).name;match=pattern.match(name)
        ground_name=item['name'].replace('/images/','/ground_truths/name_value/').removesuffix('.tif.bz2')+'.gnd'
        ground_path=BASE/finished[ground_name]['file']
        fields={k.strip():v.strip() for line in ground_path.read_text().splitlines() if '=' in line for k,v in [line.split('=',1)]}
        image=finished[item['name']]
        left=fields.get('left_eye_coords','').split();right=fields.get('right_eye_coords','').split()
        good=len(left)==len(right)==2
        if good:
            good=all(0<=float(x)<image['width'] and 0<=float(y)<image['height'] for x,y in [left,right])
        role='FA' if match.group(2) in ['fa','ba'] else 'FB'
        rows.append({'file':image['file'],'filename_subject_id':match.group(1),'official_subject_id':fields.get('id',''),
                     'candidate_role':role,'official_expression':fields.get('expression',''),
                     'pose_flag':fields.get('feret_pose_flag',''),'pose':fields.get('pose',''),
                     'lighting':fields.get('lighting',''),'date_taken':fields.get('date_taken',''),
                     'left_eye_x':left[0] if len(left)==2 else '', 'left_eye_y':left[1] if len(left)==2 else '',
                     'right_eye_x':right[0] if len(right)==2 else '', 'right_eye_y':right[1] if len(right)==2 else '',
                     'eyes_in_bounds':good,'ground_truth':finished[ground_name]['file']})
    rows.sort(key=lambda r:(r['official_subject_id'],r['file']))
    with (BASE/'fa_fb_candidates.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ids={role:{r['official_subject_id'] for r in rows if r['candidate_role']==role} for role in ['FA','FB']}
    protocol_dir=BASE/'metadata/gray_feret_cd2/partitions/by_previously_reported/feret'
    gallery=(protocol_dir/'gallery.names').read_text().splitlines()
    probes=(protocol_dir/'probe_fafb_expression.names').read_text().splitlines()
    by_stem={PurePosixPath(r['file']).stem:r for r in rows}
    assert all(name in by_stem for name in gallery+probes)
    gallery_ids={by_stem[name]['official_subject_id']:name for name in gallery}
    probe_ids={by_stem[name]['official_subject_id']:name for name in probes}
    assert len(gallery_ids)==len(gallery)==1196 and len(probe_ids)==len(probes)==1195
    assert set(probe_ids).issubset(gallery_ids)
    pairs=[]
    for name in gallery:
        row=by_stem[name];subject=row['official_subject_id']
        if subject not in probe_ids:continue
        probe=by_stem[probe_ids[subject]]
        assert row['official_expression'].upper()=='FA' and probe['official_expression'].upper()=='FB'
        pairs.append({'official_gallery_order':len(pairs)+1,'subject_id':subject,
                      'gallery_file':row['file'],'probe_file':probe['file'],
                      'gallery_ground_truth':row['ground_truth'],'probe_ground_truth':probe['ground_truth']})
    with (BASE/'official_fa_fb_pairs.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    report={'status':'candidate_download_complete','source_url':URL,'archive_etag':ETAG,
            'selection':'All original fa/ba and fb/bj image candidates, no guessed standard benchmark selection',
            'image_count':len(images),'file_count':len(records),'stored_bytes':sum(r['bytes'] for r in records),
            'role_counts':{k:sum(r['candidate_role']==k for r in rows) for k in ids},
            'official_id_counts':{k:len(v) for k,v in ids.items()},'common_official_ids':len(ids['FA']&ids['FB']),
            'eye_coordinate_valid_count':sum(r['eyes_in_bounds'] for r in rows),
            'role_annotation_mismatch_count':sum(r['candidate_role']!=r['official_expression'].upper() for r in rows),
            'filename_official_id_mismatch_count':sum(r['filename_subject_id']!=r['official_subject_id'] for r in rows),
            'full_archive_downloaded':False,'full_archive_md5_verified':False,
            'verification':'Each compressed image bzip2 CRC, TIFF pixel decoding, file SHA256; HTTP ranges and ETag checked',
            'official_gallery_image_count':len(gallery),'official_fb_probe_image_count':len(probes),
            'official_fafb_pair_count':len(pairs),'official_fafb_1195_pairing_verified':True,
            'official_pair_images_with_valid_eye_coordinates':sum(by_stem[PurePosixPath(pair[key]).stem]['eyes_in_bounds']
                                                                 for pair in pairs for key in ['gallery_file','probe_file']),
            'eye_coordinates_missing_count':sum(not all(r[k] for k in ['left_eye_x','left_eye_y','right_eye_x','right_eye_y'])
                                                for r in rows),
            'gallery_only_subject_ids':sorted(set(gallery_ids)-set(probe_ids)),
            'full_CSA_experimental_protocol_verified':False}
    (BASE/'download_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['index','candidates'])
    args=parser.parse_args()
    import base64
    if sys.stdin.isatty():
        import getpass
        credential=input('NIST username: ').strip()+':'+getpass.getpass('NIST password: ')
    else:
        credential=sys.stdin.readline().strip()
    if ':' not in credential:raise SystemExit('Supply username:password via stdin; credentials are not saved')
    AUTH='Basic '+base64.b64encode(credential.encode()).decode()
    if args.mode=='index':index_archive()
    else:download_candidates()
