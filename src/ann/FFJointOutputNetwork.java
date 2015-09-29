package ann;

import java.util.ArrayList;

/**
 * a normal feedforward network but with the final layer being fully laterally connected
 * this allows for correlation between outputs to be taken into account
 * @author alexanderbraylan
 *
 */
public class FFJointOutputNetwork extends FFNeuralNetwork {

	public FFJointOutputNetwork(ActivationFunction actFn, int numInputs,
			int numOutputs, int[] numHidden) {
		super(actFn, numInputs, numOutputs, numHidden);
		
		final ArrayList<? extends Node> outputNodes = getOutputNodes();
		for (Node n1 : outputNodes) {
			for (Node n2 : outputNodes) {
				if (n1 != n2) Connection.getOrCreate(n1, n2);
			}
		}
	}

}
