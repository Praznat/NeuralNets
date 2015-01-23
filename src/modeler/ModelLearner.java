package modeler;

import deepnets.*;

public class ModelLearner {

	private final ExperienceReplay experienceReplay;
	private final TransitionRealismAssessor modelTRA;
	private final VariableTransitionApproximator modelVTA;
	
	public ModelLearner(int errorHalfLife, int[] numHiddenTRA, int[] numHiddenVTA,
			ActivationFunction actFn, int expReplaySize) {
		this.experienceReplay = expReplaySize > 0 ? new ExperienceReplay(expReplaySize) : null;
		modelTRA = new TransitionRealismAssessor();
		modelVTA = new VariableTransitionApproximator(errorHalfLife, numHiddenVTA, actFn, expReplaySize);
	}
}
