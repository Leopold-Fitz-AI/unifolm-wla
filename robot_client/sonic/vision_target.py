"""RGB target conditioning and projection, independent of simulator object state.

This is an explicit inference/execution aid for frozen VLA, not the unassisted
baseline. Non-target colors can be desaturated while their geometry stays
visible. Robot FK/camera calibration project candidate VLA poses into RGB.
"""
import re
import itertools

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import G1Kinematics,pose_matrix

COLORS={'red':'红','green':'绿','yellow':'黄','blue':'蓝','orange':'橙'}


def command_color(instruction):
    found=[color for color,zh in COLORS.items() if re.search(r'\b'+color+r'\b',instruction.casefold()) or zh in instruction]
    if len(found)!=1:raise ValueError('RGB target selection requires exactly one explicit color')
    return found[0]


def color_mask(bgr,color):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);h,s,v=cv2.split(hsv)
    hue={'red':(h<12)|(h>168),'green':(h>35)&(h<90),
         'yellow':(h>=20)&(h<=35),'blue':(h>95)&(h<135),
         'orange':(h>=12)&(h<20)}
    if color not in hue:raise ValueError('unsupported target color')
    return hue[color]&(s>90)&(v>35)


def detect_target(bgr,color,min_area=12):
    count,labels,stats,centers=cv2.connectedComponentsWithStats(color_mask(bgr,color).astype('uint8'))
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=min_area]
    if not candidates:raise ValueError('target color is not visible')
    candidates.sort(key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)
    if len(candidates)>1 and stats[candidates[1],cv2.CC_STAT_AREA]>.4*stats[candidates[0],cv2.CC_STAT_AREA]:
        raise ValueError('multiple ambiguous target-color regions')
    i=candidates[0];x,y,w,h,area=stats[i]
    return {'color':color,'center_px':centers[i].tolist(),'bbox_px':[int(x),int(y),int(x+w),int(y+h)],'area_px':int(area)}


def condition_images(images,color):
    result={};counts={}
    for key,image in images.items():
        other=np.zeros(image.shape[:2],bool)
        for candidate in COLORS:
            if candidate!=color:other|=color_mask(image,candidate)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        conditioned=image.copy();conditioned[other]=np.repeat(gray[:,:,None],3,axis=2)[other]
        result[key]=conditioned;counts[key]=int(other.sum())
    return result,counts


class HeadProjector:
    def __init__(self,sonic_root,config,camera_mount):
        # Calibration contains robot/camera transforms only, never task objects.
        self.kin=G1Kinematics(sonic_root,config['reference_frame'])
        self.tools=[pose_matrix(p) for p in config['tool_offsets']]
        self.physical=[pose_matrix(p) for p in config.get('grasp_hold_tool_offsets',config['tool_offsets'])]
        mount=camera_mount
        self.camera=np.eye(4);self.camera[:3,3]=mount['position']
        if 'quaternion_wxyz' in mount:
            self.camera[:3,:3]=Rotation.from_quat(np.array(mount['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
        else:
            self.camera[:3,:3]=(Rotation.from_euler('y',mount['pitch_degrees'],degrees=True)*Rotation.from_quat([.5,-.5,-.5,.5])).as_matrix()
        self.fovy=float(mount['fovy'])

    def camera_to_pelvis(self,q):
        self.kin.forward(q)
        d,m=self.kin.data,self.kin.model
        transforms=[]
        for name in ('pelvis','torso_link'):
            i=m.body(name).id;t=np.eye(4);t[:3,:3]=d.xmat[i].reshape(3,3);t[:3,3]=d.xpos[i];transforms.append(t)
        return np.linalg.inv(transforms[0])@transforms[1]@self.camera

    def project(self,xyz,q,shape):
        h,w=shape[:2];camera=self.camera_to_pelvis(q)
        points=(np.asarray(xyz)-camera[:3,3])@camera[:3,:3]
        depth=-points[...,2];f=.5*h/np.tan(np.deg2rad(self.fovy/2))
        pixels=np.stack([f*points[...,0]/np.maximum(depth,1e-6)+(w-1)/2,
                         -f*points[...,1]/np.maximum(depth,1e-6)+(h-1)/2],axis=-1)
        return pixels,depth

    def action_pixels(self,action,q,hand,shape):
        side=('left','right')[hand]
        poses=np.asarray(action[f'action.{side}_ee_rpy'])[0]
        points=np.stack([(pose_matrix(p)@np.linalg.inv(self.tools[hand])@self.physical[hand])[:3,3] for p in poses])
        return self.project(points,q,shape)


class MultiViewProjector:
    def __init__(self,sonic_root,config,mounts):
        self.head=HeadProjector(sonic_root,config,mounts['unifolm_head'])
        self.mounts=mounts

    def transforms(self,q):
        head=self.head.camera_to_pelvis(q)
        wrists=self.head.kin.forward(q)
        result={'cam_left_high':head}
        for side,wrist in zip(('left','right'),wrists):
            m=self.mounts[f'unifolm_{side}_wrist'];t=np.eye(4);t[:3,3]=m['position']
            if 'quaternion_wxyz' in m:t[:3,:3]=Rotation.from_quat(np.array(m['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
            else:t[:3,:3]=(Rotation.from_euler('y',m['pitch_degrees'],degrees=True)*Rotation.from_quat([.5,-.5,-.5,.5])).as_matrix()
            result[f'cam_{side}_wrist']=wrist@t
        return result

    def locate(self,images,q,color,max_reprojection_px=8,prior_pelvis=None):
        transforms=self.transforms(q);rays=[];views=[]
        for key,image in images.items():
            try:d=detect_target(image,color)
            except ValueError:continue
            h,w=image.shape[:2];x0,y0,x1,y1=d['bbox_px']
            if min(x0,y0,w-x1,h-y1)<2:continue  # clipped cube biases its center
            camname='unifolm_head' if key=='cam_left_high' else 'unifolm_'+('left' if key=='cam_left_wrist' else 'right')+'_wrist'
            f=.5*h/np.tan(np.deg2rad(self.mounts[camname]['fovy']/2))
            pixel=np.array([(x0+x1-1)/2,(y0+y1-1)/2])
            local=np.array([(pixel[0]-(w-1)/2)/f,-(pixel[1]-(h-1)/2)/f,-1.])
            t=transforms[key];direction=t[:3,:3]@local;direction/=np.linalg.norm(direction)
            rays.append((t[:3,3],direction));views.append((key,pixel,f,h,w,t,d))
        if len(rays)<2:raise ValueError('target needs two unclipped RGB views')
        def reprojection(center):
            errors=[]
            for key,pixel,f,h,w,t,_ in views:
                local=(center-t[:3,3])@t[:3,:3]
                if local[2]>=-.02:return np.full(len(views),np.inf)
                projected=np.array([f*local[0]/-local[2]+(w-1)/2,-f*local[1]/-local[2]+(h-1)/2])
                errors.append(float(np.linalg.norm(projected-pixel)))
            return np.asarray(errors)
        hypotheses=[]
        # A hand can occlude one face and bias a color region's center. Fit all
        # view pairs and retain the largest mutually consistent subset instead
        # of allowing one occluded camera to move the target or stop every view.
        for count in range(2,len(rays)+1):
            for indices in itertools.combinations(range(len(rays)),count):
                matrices=[np.eye(3)-np.outer(rays[i][1],rays[i][1]) for i in indices]
                A=np.sum(matrices,axis=0)
                if np.linalg.cond(A)>1000:continue
                b=sum(P@rays[i][0] for P,i in zip(matrices,indices));center=np.linalg.solve(A,b)
                if not (.08<center[0]<.65 and abs(center[1])<.5 and -.15<center[2]<.5):continue
                errors=reprojection(center);inliers=errors<=max_reprojection_px
                if inliers.sum()<2:continue
                prior_error=0. if prior_pelvis is None else 100*float(np.linalg.norm(center-prior_pelvis))
                score=(-int(inliers.sum()),float(np.mean(errors[inliers]))+prior_error)
                hypotheses.append((score,center,errors,inliers))
        if not hypotheses:raise ValueError('inconsistent RGB target triangulation')
        _,center,errors,inliers=min(hypotheses,key=lambda h:h[0])
        return {'color':color,'center_pelvis':center.tolist(),'views':[v[0] for v,ok in zip(views,inliers) if ok],
                'reprojection_error_px':errors[inliers].tolist(),
                'rejected_views':[v[0] for v,ok in zip(views,inliers) if not ok],
                'detections':{v[0]:v[-1] for v in views}}

    def action_points(self,action,hand):
        side=('left','right')[hand];head=self.head
        return np.stack([(pose_matrix(p)@np.linalg.inv(head.tools[hand])@head.physical[hand])[:3,3]
                         for p in np.asarray(action[f'action.{side}_ee_rpy'])[0]])


class VisualActionSelector:
    """Choose complete frozen-VLA proposals and gate closure using RGB evidence.

    Target 3-D position is triangulated from measured RGB, never provided to the
    VLA and never used to synthesize an end-effector goal. The selected action
    pose trajectory is an unchanged member of the model's proposal batch.
    """
    def __init__(self,sonic_root,config):
        from .execution import instruction_hand
        self.options=config['visual_target']
        if not config.get('synchronized_observation'):raise ValueError('visual selection requires same-frame RGB/state')
        self.color=command_color(config['instruction']);self.hand=instruction_hand(config['instruction'])
        self.projector=MultiViewProjector(sonic_root,config,self.options['camera_mounts'])
        self.physical=G1Kinematics(sonic_root,config['reference_frame'],config.get('grasp_hold_tool_offsets',config['tool_offsets']))
        self.count=int(self.options.get('candidate_count',8))
        if not 1<=self.count<=8:raise ValueError('visual proposal count must be in [1,8]')
        self.radius=float(self.options.get('close_radius_m',.025))
        self.safe_open=np.asarray(config.get('ready_gripper_value',config['gripper_open_value']),float)
        self.closed_model=np.asarray(config['gripper_closed_value'],float)
        self.max_age=float(self.options.get('target_max_age_seconds',2.))
        if not .005<=self.radius<=.04 or not .1<=self.max_age<=3:raise ValueError('invalid visual closure bounds')
        self.target_world=None;self.last_seen=float('-inf');self.last_detection=None
        self.blocked_close_packets=0;self.selected_proposals=0;self.last_distance=None

    @staticmethod
    def base(state):
        t=np.eye(4);t[:3,:3]=Rotation.from_quat(np.asarray(state['base_quat'])[[1,2,3,0]]).as_matrix()
        t[:3,3]=state['base_position'];return t

    def observe(self,images,state,now):
        prior=None if self.target_world is None else (np.linalg.inv(self.base(state))@np.r_[self.target_world,1])[:3]
        try:detection=self.projector.locate(images,state['body_q'],self.color,prior_pelvis=prior)
        except ValueError:
            if self.target_world is None or now-self.last_seen>self.max_age:raise
            return self.last_detection
        world=(self.base(state)@np.r_[detection['center_pelvis'],1])[:3]
        if self.target_world is not None and now-self.last_seen<.5 and np.linalg.norm(world-self.target_world)>.05:
            raise ValueError('RGB target identity jumped between frames')
        self.target_world=world;self.last_seen=now;self.last_detection=detection
        return detection

    def select(self,candidates,state,now):
        if self.target_world is None or now-self.last_seen>self.max_age:raise TimeoutError('visual target stale')
        target=(np.linalg.inv(self.base(state))@np.r_[self.target_world,1])[:3]
        scores=[]
        for i,action in enumerate(candidates):
            points=self.projector.action_points(action,self.hand)
            if not np.isfinite(points).all():raise ValueError('invalid VLA proposal pose')
            distance=np.linalg.norm(points-target,axis=1)
            prefix=len(points)
            if self.options.get('select_prefix',False):
                base=self.base(state);world=points@base[:3,:3].T+base[:3,3]
                goal=self.target_world+[0,0,self.options.get('grasp_height_bias_m',.012)]
                cost=np.linalg.norm(world-goal,axis=1)
                cost+=4*np.maximum(0.,self.target_world[2]+.005-world[:,2])
                endpoint=2+int(np.argmin(cost[2:]));prefix=endpoint+1
                score=float(cost[endpoint]+.05*np.mean(cost[:prefix]))
            else:score=float(np.mean(distance[-5:])+.1*np.mean(distance))
            scores.append({'index':i,'score':score,'prefix_frames':prefix,'last_distance_m':float(distance[-1]),
                           'prefix_distance_m':float(distance[prefix-1]),'min_distance_m':float(distance.min())})
        selected=min(scores,key=lambda x:x['score'])['index'];self.selected_proposals+=1
        action=candidates[selected]
        if self.options.get('select_prefix',False):action={k:np.asarray(v)[:,:scores[selected]['prefix_frames']].copy() for k,v in action.items()}
        return action,{'selected':selected,'scores':scores,'prefix_frames':scores[selected]['prefix_frames'],'rgb_target':self.last_detection,
                                      'target_world_from_rgb':self.target_world.tolist(),'source':'frozen_vla_rgb_candidate_selection'}

    def constrain_orientation(self,action,state):
        """Keep each VLA physical TCP position; constrain only gripper orientation.

        The learned yaw is retained. Horizontal jaw closing plus a modest
        downward finger pitch avoids driving an open finger through the table.
        Original and executed EE arrays are traced separately by the client.
        """
        if 'horizontal_jaw_pitch_degrees' not in self.options:return action
        pitch=float(self.options['horizontal_jaw_pitch_degrees'])
        if not 0<=pitch<=30:raise ValueError('horizontal-jaw pitch must be in [0,30] degrees')
        result={k:np.asarray(v).copy() for k,v in action.items()};side=('left','right')[self.hand]
        base=self.base(state);head=self.projector.head
        for i,pose in enumerate(result[f'action.{side}_ee_rpy'][0]):
            physical=base@pose_matrix(pose)@np.linalg.inv(head.tools[self.hand])@head.physical[self.hand]
            yaw=np.arctan2(physical[1,0],physical[0,0])
            physical[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('y',pitch,degrees=True).as_matrix()
            ee=np.linalg.inv(base)@physical@np.linalg.inv(head.physical[self.hand])@head.tools[self.hand]
            result[f'action.{side}_ee_rpy'][0,i]=np.r_[ee[:3,3],Rotation.from_matrix(ee[:3,:3]).as_euler('xyz')]
        return result

    def gate_grip(self,sensor,grips,opened,latched,now):
        result=np.asarray(grips,float).copy()
        if latched:return result
        if self.target_world is None or now-self.last_seen>self.max_age:
            allowed=False;self.last_distance=None
        else:
            point=(self.base(sensor)@self.physical.forward(sensor['body_q'])[self.hand])[:3,3]
            self.last_distance=float(np.linalg.norm(point-self.target_world));allowed=self.last_distance<=self.radius
        # Only veto closing; never invent a close command the VLA did not give.
        direction=np.sign(opened[self.hand]-self.closed_model[self.hand])
        if not allowed and (result[self.hand]-self.safe_open[self.hand])*direction < -1e-6:
            result[self.hand]=self.safe_open[self.hand];self.blocked_close_packets+=1
        return result
