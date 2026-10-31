import json, copy, uuid, os, sys
ROOT=os.path.expandvars("$HOME/mnt/forge/ComfyUI_windows_portable/ComfyUI/user/default")
T=json.load(open(f"{ROOT}/midfr_templates.json"))
B=json.load(open(f"{ROOT}/workflows/Qwen Character Builder.json",encoding='utf-8'))
M=json.load(open(f"{ROOT}/workflows/Qwen Mannequin Edit.json",encoding='utf-8'))

ORANGE=dict(color="#b8681a",bgcolor="#4a2c10"); GREEN=dict(color="#2f7d32",bgcolor="#1f3d21")

class G:
    """node/link table with link bookkeeping. origin -10 = subgraph input node, target -20 = subgraph output node."""
    def __init__(s): s.nodes={}; s.links={}; s.nl=0; s.nin=[]; s.nout=[]
    def add(s,n):
        n=copy.deepcopy(n)
        for i in n.get('inputs',[]): i['link']=None
        for o in n.get('outputs',[]): o['links']=None
        s.nodes[n['id']]=n; return n
    def newid(s): s.nl+=1; return s.nl
    def slot(s,n,name,kind='inputs'):
        for k,x in enumerate(s.nodes[n][kind]):
            if x['name']==name: return k
        raise KeyError((n,name,kind,[x['name'] for x in s.nodes[n][kind]]))
    def unlink_target(s,t,ts):
        i=s.nodes[t]['inputs'][ts]; l=i.get('link')
        if l is not None:
            L=s.links.pop(l); o=L['origin_id']
            if o in s.nodes:
                ol=s.nodes[o]['outputs'][L['origin_slot']]['links']
                if ol and l in ol: ol.remove(l)
                if not ol: s.nodes[o]['outputs'][L['origin_slot']]['links']=None
            i['link']=None
    def link(s,o,os,t,ts,typ=None):
        if isinstance(os,str): os=s.slot(o,os,'outputs')
        if isinstance(ts,str): ts=s.slot(t,ts)
        s.unlink_target(t,ts)
        if typ is None: typ=s.nodes[o]['outputs'][os]['type'] if o in s.nodes else s.nodes[t]['inputs'][ts]['type']
        l=s.newid(); s.links[l]=dict(id=l,origin_id=o,origin_slot=os,target_id=t,target_slot=ts,type=typ)
        s.nodes[t]['inputs'][ts]['link']=l
        if o in s.nodes:
            out=s.nodes[o]['outputs'][os]
            out['links']=(out['links'] or [])+[l]
        return l
    def consumers(s,o,os):
        return [(L['target_id'],L['target_slot']) for L in s.links.values() if L['origin_id']==o and L['origin_slot']==os]
    def delete(s,i):
        for k,n in enumerate(s.nodes[i]['inputs']): s.unlink_target(i,k)
        for k,o in enumerate(s.nodes[i].get('outputs',[])):
            for (t,ts) in s.consumers(i,k): s.unlink_target(t,ts)
        del s.nodes[i]

def tpl(typ,id,pos,title=None,w=None,size=None,style=None):
    n=copy.deepcopy(T[typ]); n['id']=id; n['pos']=list(pos); n['order']=0; n['mode']=0
    if title: n['title']=title
    if w is not None: n['widgets_values']=w
    if size: n['size']=size
    if style: n.update(style)
    return n

def from_builder(i):
    return copy.deepcopy([n for n in B['nodes'] if n['id']==i][0])

def setw(n,w): n['widgets_values']=w; return n

def export_sg(g,name,inputs,outputs,wire_in,wire_out,pos_in=(-300,0),pos_out=(3000,0)):
    """inputs: [(name,type)]; wire_in: {input_index:[(node,slot)]}; wire_out: {output_index:(node,slot)}"""
    sg_in=[];sg_out=[]
    for k,(nm,ty) in enumerate(inputs):
        ids=[g.link(-10,k,t,ts,ty) for (t,ts) in wire_in[k]]
        sg_in.append(dict(id=str(uuid.uuid4()),name=nm,type=ty,linkIds=ids,localized_name=nm,pos=[pos_in[0]+44,pos_in[1]+20*k+24]))
    for k,(nm,ty) in enumerate(outputs):
        (o,os)=wire_out[k]
        l=g.link(o,os,-20,k,ty) if False else None
    # outputs: need link with target -20 (target node absent from table)
    for k,(nm,ty) in enumerate(outputs):
        (o,os)=wire_out[k]; l=g.newid()
        g.links[l]=dict(id=l,origin_id=o,origin_slot=os,target_id=-20,target_slot=k,type=ty)
        out=g.nodes[o]['outputs'][os]; out['links']=(out['links'] or [])+[l]
        sg_out.append(dict(id=str(uuid.uuid4()),name=nm,type=ty,linkIds=[l],localized_name=nm,pos=[pos_out[0]+24,pos_out[1]+20*k+24]))
    for k,i in enumerate(sg_in): i['linkIds']=[l['id'] for l in g.links.values() if l['origin_id']==-10 and l['origin_slot']==k]
    mx=max(g.nodes)
    return dict(id=str(uuid.uuid4()),version=1,state=dict(lastGroupId=0,lastNodeId=mx,lastLinkId=g.nl,lastRerouteId=0),revision=0,config={},name=name,
        inputNode=dict(id=-10,bounding=[pos_in[0],pos_in[1],128,40+20*len(inputs)]),outputNode=dict(id=-20,bounding=[pos_out[0],pos_out[1],128,40+20*len(outputs)]),
        inputs=sg_in,outputs=sg_out,widgets=[],nodes=list(g.nodes.values()),groups=[],links=list(g.links.values()),extra={})

# ------------------------------------------------------------------ Simple Edit
def build_simple():
    g=G()
    for i,p in ((334,(0,0)),(335,(0,150)),(336,(0,300)),(338,(350,0)),(342,(350,150)),(344,(800,0)),(345,(1150,0))):
        n=from_builder(i); n['pos']=list(p); g.add(n)
    g.nodes[342]['widgets_values']=['', 'deformed, extra limbs, blurry, low quality', 1024]
    g.add(tpl('StepPreview',400,(1450,0),'Result',['Simple edit result']))
    g.link(334,0,338,0); g.link(338,0,344,0)
    g.link(335,0,342,'clip'); g.link(336,0,342,'vae')
    g.link(342,0,344,'positive'); g.link(342,1,344,'negative'); g.link(342,2,344,'latent_image')
    g.link(344,0,345,0); g.link(336,0,345,'vae'); g.link(345,0,400,0)
    g.add(tpl('QwenInputsUnpack',430,(-700,300),'Qwen Inputs (unpack)'))
    g.link(-10,2,430,'inputs','QWEN_INPUTS')
    for k in (1,2,3): g.link(430,f'ref_{k}',342,f'images.image_{k+1}','IMAGE')
    return export_sg(g,"Simple Edit",[("image","IMAGE"),("prompt","STRING"),("inputs","QWEN_INPUTS")],[("IMAGE","IMAGE")],
        {0:[(342,g.slot(342,'images.image_1'))],1:[(342,g.slot(342,'prompt'))],2:[]},{0:(400,0)})

# ------------------------------------------------------------------ Advanced Edit (existing Controlled Edit subgraph)
def build_advanced():
    sg=copy.deepcopy(M['definitions']['subgraphs'][0]); sg['name']="Advanced Edit"; sg['id']=str(uuid.uuid4())
    labels={11:"Regions (skin / clothing)",24:"Mannequin (shaded)",15:"Clothed depth",26:"Control depth (to Qwen)"}
    for n in sg['nodes']:
        if n['id'] in labels:
            lk=n['inputs'][0]['link']
            new=tpl('StepPreviewSink',n['id'],n['pos'],n.get('title'),[labels[n['id']]]); new['inputs'][0]['link']=lk; new['title']=labels[n['id']]
            n.clear(); n.update(new)
    def raw_link(o,os,t,ts,typ):
        lid=max(l['id'] for l in sg['links'])+1
        sg['links'].append(dict(id=lid,origin_id=o,origin_slot=os,target_id=t,target_slot=ts,type=typ))
        nd=lambda i:[n for n in sg['nodes'] if n['id']==i][0]
        nd(t)['inputs'][ts]['link']=lid
        if o>0:
            out=nd(o)['outputs'][os]; out['links']=(out.get('links') or [])+[lid]
        return lid
    un=tpl('QwenInputsUnpack',100,(-700,300),'Qwen Inputs (unpack)'); sg['nodes'].append(un)
    l0=raw_link(-10,2,100,0,'QWEN_INPUTS')
    sg['inputs'].append(dict(id=str(uuid.uuid4()),name='inputs',type='QWEN_INPUTS',linkIds=[l0],localized_name='inputs',pos=[-84,957.5]))
    enc=[n for n in sg['nodes'] if n['id']==31][0]
    for k in (1,2,3):
        ts=[i['name'] for i in enc['inputs']].index(f'images.image_{k+1}'); os_=[o['name'] for o in un['outputs']].index(f'ref_{k}')
        raw_link(100,os_,31,ts,'IMAGE')
    sg['state']['lastNodeId']=max(sg['state']['lastNodeId'],100); sg['state']['lastLinkId']=max(l['id'] for l in sg['links'])
    return sg

# ------------------------------------------------------------------ Model Extraction and Edit
def build_extract():
    g=G()
    for n in B['nodes']: g.nodes[n['id']]=copy.deepcopy(n)
    for n in g.nodes.values():
        for i in n.get('inputs',[]): i['link']=None
        for o in n.get('outputs',[]): o['links']=None
    for l in B['links']:
        lid,o,os,t,ts,ty=l[:6]
        g.links[lid]=dict(id=lid,origin_id=o,origin_slot=os,target_id=t,target_slot=ts,type=ty)
        g.nodes[t]['inputs'][ts]['link']=lid
        out=g.nodes[o]['outputs'][os]; out['links']=(out['links'] or [])+[lid]
    g.nl=max(g.links)
    g.add(tpl('QwenInputsUnpack',430,(-700,0),'Qwen Inputs (unpack)'))
    g.link(-10,0,430,'inputs','QWEN_INPUTS')
    # drop unused / moved nodes
    srcL=g.links[g.nodes[303]['inputs'][0]['link']]; msrc=(srcL['origin_id'],srcL['origin_slot'])
    cons=g.consumers(303,0); g.delete(303)
    for (t,ts) in cons: g.link(msrc[0],msrc[1],t,ts)
    for i in (349,350,364,365,366): g.delete(i)
    # source photo -> subgraph input 0
    for (t,ts) in g.consumers(324,0): g.link(430,'image',t,ts,'IMAGE')
    g.delete(324)
    # preview nodes -> in-line step previews (splice into the chain)
    lab={302:"Subject cut-out",347:"A-pose redraw",348:"A-pose control depth",358:"Generated views",361:"Textured front view"}
    for pid,l in lab.items():
        old=g.nodes[pid]; srcL=g.links[old['inputs'][0]['link']]; src=(srcL['origin_id'],srcL['origin_slot'])
        outs=[(t,ts) for (t,ts) in g.consumers(pid,0)] if old['outputs'][0].get('links') else []
        # consumers of the source (other than the preview itself) that should see the previewed image
        others=[(t,ts) for (t,ts) in g.consumers(*src) if t!=pid] if pid!=302 else []
        pos=old['pos']; g.delete(pid)
        n=g.add(tpl('StepPreview',pid,pos,None,[l])); n['title']=l
        g.link(src[0],src[1],pid,0)
        for (t,ts) in (outs if pid==302 else others): g.link(pid,0,t,ts)
    # CharacterSave: lazy + skip
    cs=g.nodes[363]; pos=cs['pos']
    g.delete(363); g.add(tpl('CharacterSave',363,pos,None,['character',False,False]))
    g.link(359,0,363,'mesh'); g.link(362,0,363,'character'); g.link(347,0,363,'source_image'); g.link(361,0,363,'thumbnail')
    g.link(430,'name',363,'name','STRING'); g.link(430,'use_model',363,'skip','BOOLEAN')
    # edit stage
    x=7000
    g.add(tpl('QwenLazyPick',400,(x,0),'Mesh: built or from library')); g.nodes[400]['widgets_values']=[False]
    g.add(tpl('QwenLazyPick',401,(x,150),'Reference: A-pose or library source')); g.nodes[401]['widgets_values']=[False]
    g.add(tpl('MeshOrbitCameras',402,(x,300),'Front depth of the character mesh',[1,5,28,576,864,1.1,False,T['MeshOrbitCameras']['widgets_values'][7],0,0.3,T['MeshOrbitCameras']['widgets_values'][10]],size=[400,300]))
    for i,p in ((354,(x,700)),(355,(x+450,0)),(344,(x+900,0)),(345,(x+1250,0))):
        n=from_builder(i); n['pos']=list(p); n['id']={354:403,355:404,344:405,345:406}[i]; g.add(n)
    g.nodes[403]['widgets_values']=['', 'deformed, extra limbs, blurry, low quality', 704]
    g.nodes[403]['title']='Edit prompt'; g.nodes[404]['title']='Character depth control'
    g.add(tpl('StepPreview',407,(x+1550,0),'Edited result',['Edited with the character']))
    g.add(tpl('StringConcatenate',408,(x,1100),'Prompt + keep-identity suffix',['','Keep exactly the same face, hairstyle shape, body proportions and outfit as <image1> unless the instruction changes them. Plain light grey studio background unless the instruction says otherwise.',' ']))
    g.add(tpl('StringConcatenate',409,(x-500,1100),'Prefix + prompt',['Redraw the character from <image1> as a full-body photograph, matching the depth guide exactly. Apply this change:','',' ']))
    g.link(-10,1,409,'string_b','STRING'); g.link(409,0,408,'string_a','STRING')
    g.link(430,'use_model',400,'use_b','BOOLEAN'); g.link(359,0,400,'a','MESH'); g.link(430,'mesh',400,'b','MESH')
    g.link(430,'use_model',401,'use_b','BOOLEAN'); g.link(347,0,401,'a','IMAGE'); g.link(430,'image',401,'b','IMAGE')
    g.add(tpl('QwenLazyPick',410,(x,1300),'Character pose data: built or library')); g.nodes[410]['widgets_values']=[False]
    g.link(430,'use_model',410,'use_b','BOOLEAN'); g.link(362,0,410,'a','MHR_POSE_DATA'); g.link(430,'character',410,'b','MHR_POSE_DATA')
    n=from_builder(332); n['id']=411; n['pos']=[x,1500]; n['title']='Pose of the reference image'; g.add(n)
    g.link(331,0,411,'sam3d_body_model'); g.link(430,'pose_image',411,'image','IMAGE')
    g.add(tpl('QwenLazyPick',414,(x+450,1600),'Pose: from image or saved')); g.nodes[414]['widgets_values']=[False]
    g.link(430,'use_saved',414,'use_b','BOOLEAN'); g.link(411,0,414,'a','MHR_POSE_DATA'); g.link(430,'saved_pose',414,'b','MHR_POSE_DATA')
    g.add(tpl('PoseSave',415,(x+450,1800),'Save pose to library',['pose',True]))
    g.link(411,0,415,'pose','MHR_POSE_DATA'); g.link(430,'pose_image',415,'image','IMAGE'); g.link(430,'pose_name',415,'name','STRING'); g.link(430,'skip_pose_save',415,'skip','BOOLEAN')
    g.add(tpl('MeshRigPose',412,(x+450,1300),'Re-pose the character (MHR rig)',[True,True,40,4]))
    g.link(331,0,412,'sam3d_body_model'); g.link(400,0,412,'mesh','MESH'); g.link(410,0,412,'character'); g.link(414,0,412,'reference')
    g.add(tpl('QwenLazyPick',413,(x+900,1300),'Mesh: A-pose or re-posed')); g.nodes[413]['widgets_values']=[False]
    g.link(430,'use_pose',413,'use_b','BOOLEAN'); g.link(400,0,413,'a','MESH'); g.link(412,0,413,'b','MESH')
    g.link(413,0,402,'mesh','MESH')
    g.link(335,0,403,'clip'); g.link(336,0,403,'vae'); g.link(401,0,403,'images.image_1','IMAGE'); g.link(408,0,403,'prompt','STRING')
    g.link(338,0,404,'model'); g.link(337,0,404,'model_patch'); g.link(336,0,404,'vae'); g.link(402,0,404,'image')
    g.link(404,0,405,'model'); g.link(403,0,405,'positive'); g.link(403,1,405,'negative'); g.link(403,2,405,'latent_image')
    g.link(405,0,406,'samples'); g.link(336,0,406,'vae'); g.link(406,0,407,0)

    # ---- imported models: auto rig (SAM 3D Body on a render) and auto texture (multi-view Qwen from the reference image)
    n=from_builder(360); n['id']=416; n['pos']=[x,2100]; n['title']='Render imported model (for rig)'; g.add(n)
    g.link(430,'mesh',416,'mesh','MESH')
    n=from_builder(332); n['id']=417; n['pos']=[x+450,2100]; n['title']='Pose of the imported model (auto-rig)'; g.add(n)
    g.link(331,0,417,'sam3d_body_model'); g.link(416,0,417,'image','IMAGE')
    g.add(tpl('QwenLazyPick',418,(x+900,2100),'Character data: stored or auto-rigged')); g.nodes[418]['widgets_values']=[False]
    g.link(430,'needs_rig',418,'use_b','BOOLEAN'); g.link(430,'character',418,'a','MHR_POSE_DATA'); g.link(417,0,418,'b','MHR_POSE_DATA')
    g.link(418,0,410,'b','MHR_POSE_DATA')
    n=from_builder(351); n['id']=423; n['pos']=[x,2400]; n['title']='Decimate imported model'; g.add(n)
    n=from_builder(352); n['id']=419; n['pos']=[x+450,2400]; n['title']='Unwrap imported model'; g.add(n)
    g.link(430,'mesh',423,'mesh','MESH'); g.link(423,0,419,'mesh','MESH')
    g.add(tpl('QwenLazyPick',421,(x+900,2400),'Mesh to texture: built or imported')); g.nodes[421]['widgets_values']=[False]
    g.link(430,'use_model',421,'use_b','BOOLEAN'); g.link(352,0,421,'a','MESH'); g.link(419,0,421,'b','MESH')
    g.link(421,0,353,'mesh','MESH'); g.link(421,0,359,'mesh','MESH')
    g.add(tpl('QwenLazyPick',422,(x+900,2650),'Texturing reference: A-pose or input image')); g.nodes[422]['widgets_values']=[False]
    g.link(430,'use_model',422,'use_b','BOOLEAN'); g.link(347,0,422,'a','IMAGE'); g.link(430,'image',422,'b','IMAGE')
    g.link(422,0,354,'images.image_1','IMAGE')
    g.add(tpl('QwenLazyPick',420,(x+1350,2400),'Library mesh: stored or freshly textured')); g.nodes[420]['widgets_values']=[False]
    g.link(430,'needs_texture',420,'use_b','BOOLEAN'); g.link(430,'mesh',420,'a','MESH'); g.link(359,0,420,'b','MESH')
    g.link(420,0,400,'b','MESH')
    # ---- same-character angles feed the A-pose and view-texturing stages; reference images feed the edit
    for k in (1,2,3):
        g.link(430,f'angle_{k}',342,f'images.image_{k+1}','IMAGE'); g.link(430,f'angle_{k}',354,f'images.image_{k+1}','IMAGE')
        g.link(430,f'ref_{k}',403,f'images.image_{k+1}','IMAGE')
    g.add(tpl('StringConcatenate',424,(x,2800),'A-pose prompt + other angles',['','',' ']))
    g.link(341,0,424,'string_a','STRING'); g.link(430,'angle_prompt',424,'string_b','STRING'); g.link(424,0,342,'prompt','STRING')
    g.add(tpl('StringConcatenate',425,(x+450,2800),'View prompt + other angles',['','',' ']))
    g.link(353,'view_prompt',425,'string_a','STRING'); g.link(430,'angle_prompt',425,'string_b','STRING'); g.link(425,0,354,'prompt','STRING')
    # the A-pose stage only needs to run when building: nothing else to do (lazy)
    sg=export_sg(g,"Model Extraction and Edit",[("inputs","QWEN_INPUTS"),("prompt","STRING")],
        [("image","IMAGE"),("model","MESH")],{0:[],1:[]},{0:(407,0),1:(413,0)},pos_in=(-300,0),pos_out=(x+2000,0))
    # the sg inputs were already linked above via link(-10,...); rebuild linkIds from the link table
    for k,i in enumerate(sg['inputs']): i['linkIds']=[l['id'] for l in sg['links'] if l['origin_id']==-10 and l['origin_slot']==k]
    return sg

NOTE_TEXT = """## Required models
Put each file in the folder shown (inside `ComfyUI/models/`). Names must match exactly.

**Qwen Image 2.1 (all modes)**
| File | Folder |
|---|---|
| `qwen_image_2.1_turbo_int8_convrot.safetensors` | `diffusion_models/` |
| `qwen3vl_8b_w4a8.safetensors` | `text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | `vae/` |

**Advanced edit** (planner, segmentation, depth, body)
| File | Folder |
|---|---|
| `Qwen-Image-2.1-Fun-Controlnet-Union-ComfyUI.safetensors` | `model_patches/` |
| `Qwen3-VL-4B-Instruct-heretic-7refusal` (folder) | `LLM/` |
| `sapiens2_1b_seg.safetensors` | `sapiens2/` (from [facebook/sapiens2-seg-1b](https://huggingface.co/facebook/sapiens2-seg-1b)) |
| `sapiens2_1b_pointmap.safetensors` | `sapiens2/` (from [facebook/sapiens2-pointmap-1b](https://huggingface.co/facebook/sapiens2-pointmap-1b)) |
| `sam_3d_body_dinov3_int8_convrot.safetensors` | `detection/` |

**Model extraction** (also needs the SAM 3D Body + Sapiens2 files above and the control patch)
| File | Folder |
|---|---|
| `pixal3d_int8_convrot.safetensors` | `diffusion_models/` |
| `trellis_2_shape_vae_bf16.safetensors` | `vae/` |
| `dino_v3_L_naf_fp32.safetensors` | `clip_vision/` |
| `moge_2_vitl_normal_fp16.safetensors` | `geometry_estimation/` |
| `birefnet.safetensors` | `background_removal/` |

Only the two Sapiens2 download sources are confirmed (in that node pack's README). For the other files, search the exact filename on Hugging Face; this workflow could not verify a source for them.

**Custom node packs:** `midfr-comfyui-nodes` (this repo), `ComfyUI-Sapiens2`, plus the SAM 3D Body / Trellis2 / Pixal3D / MoGe node support that ships with ComfyUI.

**Created by the workflow:** `models/characters/` (saved 3D characters and imported models), `models/poses/` (saved poses).
"""

def main():
    sgs=[build_simple(),build_advanced(),build_extract()]
    top=G(); top.nl=0
    inst={}
    def sg_node(id,sg,pos,mode_idx,title):
        n=dict(id=id,type=sg['id'],pos=list(pos),size=[330,140],flags={},order=0,mode=0,
               inputs=[dict(name=i['name'],type=i['type'],link=None,localized_name=i['name']) for i in sg['inputs']],
               outputs=[dict(name=o['name'],type=o['type'],links=None,localized_name=o['name']) for o in sg['outputs']],
               properties={'midfr_mode':mode_idx},title=title,**GREEN)
        return n
    top.add(tpl('QwenModeSelect',1,(0,0),'Mode Selection',['Advanced edit'],style=ORANGE))
    top.add(tpl('QwenInputs',2,(0,200),'Qwen Inputs',['{"image":"train_src.png"}'],style=ORANGE,size=[430,420]))
    top.add(tpl('PrimitiveStringMultiline',3,(0,-260),'Prompt',['change her hair color to red'],size=[340,160]))
    top.add(sg_node(10,sgs[0],(520,0),0,'Simple Edit'))
    top.add(sg_node(11,sgs[1],(520,260),1,'Advanced Edit'))
    top.add(sg_node(12,sgs[2],(520,520),2,'Model Extraction and Edit'))
    top.add(tpl('QwenOutputSelect',20,(1000,200),'Output',style=ORANGE))
    top.add(tpl('SaveImage',21,(1300,100),'Image Output',['%date:yyyy-MM-dd%/QWEN/Pipeline']))
    top.add(tpl('MeshToFile3D',22,(1300,500),None))
    top.add(tpl('Save3DAdvanced',23,(1600,500),'3D model preview / save',['3d/character']))
    top.add(tpl('MidfrLivePreview',30,(1950,0),'Live Preview',style=ORANGE))
    nt=from_builder(366); nt['id']=40; nt['pos']=[-980,0]; nt['size']=[900,1180]; nt['title']='Required models'; nt['widgets_values']=[NOTE_TEXT]
    nt['inputs']=nt.get('inputs') or []; nt['outputs']=nt.get('outputs') or []
    top.nodes[40]=nt
    # links
    top.link(3,0,2,'prompt','STRING')
    for t in (10,11): top.link(2,1,t,0,'IMAGE')
    for t in (10,11,12): top.link(2,2,t,1,'STRING')
    top.link(2,0,10,2,'QWEN_INPUTS'); top.link(2,0,11,2,'QWEN_INPUTS'); top.link(2,0,12,0,'QWEN_INPUTS')
    top.link(10,0,20,'simple'); top.link(11,0,20,'advanced'); top.link(12,0,20,'extracted'); top.link(12,1,20,'model')
    top.link(20,0,21,'images'); top.link(20,1,22,0)
    top.link(22,0,23,0)
    wf=dict(id=str(uuid.uuid4()),revision=0,last_node_id=max(top.nodes),last_link_id=top.nl,nodes=list(top.nodes.values()),
            links=[[l['id'],l['origin_id'],l['origin_slot'],l['target_id'],l['target_slot'],l['type']] for l in top.links.values()],
            groups=[],definitions=dict(subgraphs=sgs),config={},extra=dict(ds=dict(scale=0.6,offset=[100,100])),version=0.4)
    return wf
if __name__=="__main__":
    wf=main()
    out=sys.argv[1]; json.dump(wf,open(out,'w',encoding='utf-8'))
    print('written',out,len(wf['nodes']),[ (s['name'],len(s['nodes']),len(s['links'])) for s in wf['definitions']['subgraphs']])
