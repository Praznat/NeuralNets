package modeler;

import java.util.*;

import ann.*;

public class JointDistributionModeler extends ModelNeuralNet {

	protected boolean hasFamiliarityNode = false;
	protected boolean shouldDisconnect = true;
	
	protected JointDistributionModeler(ActivationFunction actFn, int[] numHidden, int errorHalfLife) {
		super(actFn, numHidden, errorHalfLife);
		if (numHidden.length > 1) throw new IllegalStateException("JDM can only have 1 hidden layer"
				+ " else disconnecting self-conditions is too hard (but doable just annoying)");
	}

	@Override
	protected void analyzeTransition(TransitionMemory tm, double lRate,
			double mRate, double sRate) {
		final double[] ins = tm.getAllVars();
		final double[] postState = tm.getPostState();
		final double[] targets = new double[postState.length + (hasFamiliarityNode ? 1 : 0)];
		System.arraycopy(postState, 0, targets, 0, postState.length);
		if (hasFamiliarityNode) targets[postState.length] = 1;
		nnLearn(ins, targets, lRate, mRate, sRate);
	}

	@Override
	public void learn(Collection<TransitionMemory> memories, double stopAtErrThreshold,
			long displayProgressMs, int iterations, double lRate, double mRate, double sRate,
			boolean isRecordingTraining, ArrayList<Double> trainingErrorLog) {
		disconnectSelfConditions(memories.iterator().next());
		super.learn(memories, stopAtErrThreshold, displayProgressMs, iterations,
				lRate, mRate, sRate, isRecordingTraining, trainingErrorLog);
	}
	
	private void disconnectSelfConditions(TransitionMemory tm) {
		if (!shouldDisconnect) return;
		int avlen = tm.getAllVars().length;
		int pslen = tm.getPostState().length;
		int postStatesIndex = avlen - pslen;
		adjustNNSize(avlen, pslen + (hasFamiliarityNode ? 1 : 0));
		ArrayList<? extends Node> ins = ann.getInputNodes();
		ArrayList<? extends Node> outs = ann.getOutputNodes();
		ann.getLayers().get(0).setName("I");
		ann.getLayers().get(1).setName("H");
		ann.getLayers().get(2).setName("O");
		ArrayList<? extends Node> hidden = ann.getLayers().get(1).getNodes();
		int numHidden = hidden.size();
		int nvars = outs.size() - (hasFamiliarityNode ? 1 : 0);
		int hPerGroup = (int) Math.round(((double) numHidden) / nvars);
		if (hPerGroup == 0) throw new IllegalStateException("you need more hidden nodes than variables");
		Map<Node, Integer> hiddenGroups = new HashMap<Node, Integer>();
		for (int i = 0; i < numHidden; i++) hiddenGroups.put(hidden.get(i), i / hPerGroup);
		Collection<Connection> disconnections = new ArrayList<Connection>();
		// kill input to same hidden
		for (int i = postStatesIndex; i < ins.size(); i++) {
			Node input = ins.get(i);
			int designatedGroup = i - postStatesIndex;
			for (Connection conn : input.getOutputConnections())
				if (hiddenGroups.get(conn.getOutputNode()) == designatedGroup) disconnections.add(conn);
		}
		// kill output from different hidden
		for (int i = 0; i < pslen; i++) {
			Node output = outs.get(i);
			int designatedGroup = i;
			for (Connection conn : output.getInputConnections()) {
				Node hNode = conn.getInputNode();
				if (BiasNode.isBias(hNode)) continue;
				if (hiddenGroups.get(hNode) != designatedGroup) disconnections.add(conn);
			}
		}
		for (Connection conn : disconnections) {
			conn.getInputNode().getOutputConnections().remove(conn);
			conn.getOutputNode().getInputConnections().remove(conn);
//			System.out.println("Removing connection " + conn);
		}
//		System.out.println("Same var connections removed");
	}

	public void toggleShouldDisconnect(boolean b) {
		shouldDisconnect = b;
	}
	
	@Override
	public void setANN(FFNeuralNetwork ann) {
		super.setANN(ann);
		toggleShouldDisconnect(false);
	}
}
