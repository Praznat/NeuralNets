package modulemanagement;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import ann.ActivationFunction;
import ann.Connection;
import ann.ExperienceReplay;
import ann.FFNeuralNetwork;
import ann.Node;
import ann.Utils;
import ann.indirectencodings.IndirectInput;
import ann.indirectencodings.RelationManager;
import modeler.TransitionMemory;

/**
 * Given the key of some output variable, it looks up the keys to related input variables.
 * Then, given the values of those inputs for some sample, it generates the value for the output.
 *
 */
@SuppressWarnings("serial")
public abstract class ReusableModule<T> implements Serializable {
	
	protected final Collection<IndirectInput> relations = new ArrayList<IndirectInput>();
	protected final Map<IndirectInput, Node> rel2ModuleInput = new HashMap<IndirectInput, Node>();
	protected FFNeuralNetwork neuralNet;
	
	private int[] getInKeys(T outputOfInterest, RelationManager<T> relMngr) {
		ArrayList<T> inputsOfInterest = getRelInputs(outputOfInterest, relMngr);
		final int inN = inputsOfInterest.size();
		int[] inKeys = new int[inN];
		int i = 0;
		for (T t : inputsOfInterest) inKeys[i++] = getVectorKey(t);
		return inKeys;
	}
	
	/**
	 * trains new or existing neural net representing the behavior of this module
	 */
	protected void trainModuleNN(ExperienceReplay<TransitionMemory> experienceReplay, T outputOfInterest,
			RelationManager<T> relMngr, int[] numHidden, int epochs, double lRate, double mRate, double sRate) {

		int outKey = getVectorKey(outputOfInterest);
		int[] inKeys = getInKeys(outputOfInterest, relMngr);
		
		for(int i = 0; i < epochs; i++) {
			Iterator<TransitionMemory> iter = experienceReplay.getBatch().iterator();
			double inActivations[] = new double[inKeys.length];
//			double err = 0;
			while (iter.hasNext()) {
				TransitionMemory tm = iter.next();
				double[] preStateAndAction = tm.getPreStateAndAction();
				double[] postState = tm.getPostState();
				int j = 0;
				for (int k : inKeys) inActivations[j++] = k >= 0 ? preStateAndAction[k] : 0;
				if (neuralNet == null) neuralNet = new FFNeuralNetwork(ActivationFunction.SIGMOID0p5,
						inActivations.length, 1, numHidden);
				FFNeuralNetwork.feedForward(neuralNet.getInputNodes(), inActivations);
				FFNeuralNetwork.backPropagate(neuralNet.getOutputNodes(), lRate, mRate, sRate, postState[outKey]);
//				err += FFNeuralNetwork.getError(new double[] {postState[outKey]}, neuralNet.getOutputNodes());
			}
//			System.out.println(i + "	" + err / experienceReplay.getSize());
		}
//		System.out.println();
		if (rel2ModuleInput.isEmpty()) {
			ArrayList<? extends Node> inputNodes = neuralNet.getInputNodes();
			int j = 0;
			for (IndirectInput rel : relations) rel2ModuleInput.put(rel, inputNodes.get(j++));
		}
	}
	
	protected abstract int getVectorKey(T key);

	/**
	 * returns an output for the given input through this module's ANN
	 */
	public double evaluateOutput(T outputOfInterest, RelationManager<T> relMngr, double[] fullInput) {
		if (neuralNet == null) throw new IllegalStateException("train before querying");
		int[] inKeys = getInKeys(outputOfInterest, relMngr);
		final double[] activations = new double[inKeys.length];
		int i = 0;
		for (int k : inKeys) activations[i++] = k >= 0 ? fullInput[k] : 0;
		FFNeuralNetwork.feedForward(neuralNet.getInputNodes(), activations);
		final double result = neuralNet.getOutputNodes().get(0).getActivation();
		return result;
	}
	
	protected ArrayList<T> getRelInputs(T output, RelationManager<T> relMngr) {
		ArrayList<T> relNodes = new ArrayList<T>();
		for (IndirectInput rel : relations) {
			T relNode = (T) relMngr.getRelNode(output, rel);
			relNodes.add(relNode);
		}
		return relNodes;
	}
	
	@Override
	public String toString() {
		if (rel2ModuleInput.isEmpty()) return "uninitialized module";
		StringBuilder sb = new StringBuilder();
		for (IndirectInput ii : relations) {
			Node node = rel2ModuleInput.get(ii);
			if (node == null) continue;
			Collection<Connection> outputConnections = node.getOutputConnections();
			if (!outputConnections.isEmpty()) {
				sb.append(ii.toString().substring(0, 1));
				sb.append(Utils.round(outputConnections.iterator().next().getWeight().getWeight(), 2));
				sb.append(" ");
			}
		}
		return sb.toString();
	}

	public FFNeuralNetwork getNeuralNet() {
		return neuralNet;
	}
}
