"""Object-independent constraints for a stationary single-arm instruction."""
import numpy as np

from .adapter import pose_to_rpy


def instruction_hand(instruction):
    command = instruction.casefold()
    left = 'left hand' in command or '左手' in command
    right = 'right hand' in command or '右手' in command
    if left == right:
        raise ValueError('stationary grasp requires an explicit left/right hand instruction')
    return 0 if left else 1


class StationaryArmExecution:
    """Preserve VLA's selected EE goals; hold waist and the unused arm.

    The anchor is the measured ready posture, not a task/object waypoint.
    Apply before IK so the selected EE is solved at the held waist angles.
    """
    def __init__(self, kin, q, grippers, instruction, height, waist_reference=None):
        self.kin=kin
        self.hand = instruction_hand(instruction)
        self.other = 1-self.hand
        self.anchor = np.asarray(q, float).copy()
        if waist_reference is not None:self.anchor[12:15]=waist_reference
        self.grip = float(grippers[self.other])
        self.pose = pose_to_rpy(kin.forward(self.anchor)[self.other])
        self.height = float(height)
        self.waist_rejections=0

    def compensate_waist(self, nominal, measured, observation_base=None, current_base=None):
        """Keep the selected VLA EE goal while the waist has tracking error."""
        targets=self.kin.forward(nominal)
        if observation_base is not None:
            targets[self.hand]=np.linalg.inv(current_base)@observation_base@targets[self.hand]
        seed=np.asarray(nominal,float).copy()
        seed[:15]=measured[:15]
        other_start=15+7*self.other
        # The unused arm stays in its captured joint posture at the actual waist.
        targets[self.other]=self.kin.forward(seed)[self.other]
        try:
            result=self.kin.solve(targets,seed,position_tolerance=.01,rotation_tolerance=.05)
        except ValueError as exc:
            if not str(exc).startswith('IK target unreachable'):raise
            # Near an arm workspace boundary, retain the already validated
            # nominal joint goal rather than inventing an unreachable correction.
            self.waist_rejections+=1
            return np.asarray(nominal,float).copy()
        result[:15]=nominal[:15]
        result[other_start:other_start+7]=nominal[other_start:other_start+7]
        return result

    def constrain(self, action):
        result = {k:np.asarray(v).copy() for k,v in action.items()}
        side = ('left','right')[self.other]
        result['action.lower_body'][:,:,12:15] = self.anchor[12:15]
        result[f'action.{side}_ee_rpy'][:,:,:] = self.pose
        result[f'action.{side}_gripper'][:,:,:] = self.grip
        result['action.base_command'][:,:,:3] = 0
        result['action.base_command'][:,:,3] = self.height
        return result

    def freeze_unused_joints(self, reference):
        # Redundant IK may change null-space posture without moving the unused
        # end effector. Preserve the measured unused arm joint angles exactly.
        start = 15+7*self.other
        # The seed was measured at a different waist angle. Start the nominal
        # trajectory from its first actual model goal; physical output limiting
        # occurs after conversion back to measured waist/base coordinates.
        reference.q[0]=reference.q[1].copy()
        reference.q[:,12:15]=self.anchor[12:15]
        reference.q[:,start:start+7] = self.anchor[start:start+7]
        return reference
