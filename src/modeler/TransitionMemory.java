package modeler;

import deepnets.ExperienceReplay.Memory;
import deepnets.*;

public class TransitionMemory implements Memory {

	protected double[] preStateVars = {};
	protected double[] action = {};
	protected double[] postStateVars = {};
	
	public TransitionMemory(double[] preStateVars, double[] action, double[] postStateVars) {
		this.preStateVars = preStateVars.clone();
		this.action = action != null ? action.clone() : new double[] {};
		if (postStateVars != null) this.postStateVars = postStateVars.clone();
	}
	public TransitionMemory(double[] preStateAndAction, int numStateVars, double[] postState) {
		preStateVars = new double[numStateVars];
		action = new double[preStateAndAction.length - numStateVars];
		if (postState != null) postStateVars = postStateVars.clone();
		System.arraycopy(preStateAndAction, 0, preStateVars, 0, numStateVars);
		System.arraycopy(preStateAndAction, numStateVars, action, 0, action.length);
	}
	
	public double[] getPreStateAndAction() {
		double[] result = new double[preStateVars.length + action.length];
		System.arraycopy(preStateVars, 0, result, 0, preStateVars.length);
		System.arraycopy(action, 0, result, preStateVars.length, action.length);
		return result;
	}

	public double[] getAllVars() {
		double[] result = new double[preStateVars.length + action.length + postStateVars.length];
		System.arraycopy(preStateVars, 0, result, 0, preStateVars.length);
		System.arraycopy(action, 0, result, preStateVars.length, action.length);
		System.arraycopy(postStateVars, 0, result, preStateVars.length + action.length, postStateVars.length);
		return result;
	}
	
	public double[] getPostState() {
		double[] result = new double[postStateVars.length];
		System.arraycopy(postStateVars, 0, result, 0, postStateVars.length);
		return result;
	}
	
	@Override
	public String toString() {
		return "	S:	" + Utils.stringArray(preStateVars, 2) + "	A:	" + Utils.stringArray(action, 2) 
				+ "	S':	" + Utils.stringArray(postStateVars, 2);
	}

}
