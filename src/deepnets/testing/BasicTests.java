package deepnets.testing;

import java.util.*;

import modeler.*;

import reasoner.Foresight;
import utils.RandomUtils;

import deepnets.*;

public class BasicTests {

	public static void main(String[] args) {
		xor();
		modeling();
		bayesian();
		envtranslator();
		transitionApproximator();
		fakePole();
	}

	private static void xor() {
		System.out.println("XOR");
		Layer<Node> inputLayer = Layer.createInputLayer(2, Node.BASIC_NODE_FACTORY);
		Layer<Node> hiddenLayer = Layer.createHiddenFromInputLayer(inputLayer, 2,
				ActivationFunction.SUPERSIGMOID, Node.BASIC_NODE_FACTORY);
		BiasNode.connectToLayer(hiddenLayer);
		Layer<Node> outputLayer = Layer.createHiddenFromInputLayer(hiddenLayer, 1,
				ActivationFunction.SUPERSIGMOID, Node.BASIC_NODE_FACTORY);
		BiasNode.connectToLayer(outputLayer);
		double[][] inputSamples = {{0,0},{0,1},{1,0},{1,1}};
		System.out.println("Untrained");
		for (double[] is : inputSamples) {
			FFNeuralNetwork.feedForward(inputLayer.getNodes(), is);
			String output = "";
			for (Node n : outputLayer.getNodes()) output += n.getActivation() + " ";
			System.out.println(output);
		}
		
		double[][] outputSamples = {{0},{1},{1},{0}};
		
		Collection<DataPoint> data = DataPoint.createData(inputSamples, outputSamples);
		System.out.println("E: " + FFNeuralNetwork.stdError(inputLayer.getNodes(), outputLayer.getNodes(), data));
		
		ControlPanel.learnFromBackPropagation(inputLayer.getNodes(), outputLayer.getNodes(), data, 10000,
				2, 0.8, 0.5, 0.3, 0.05, 0.01);

		System.out.println("Trained");
		for (double[] is : inputSamples) {
			FFNeuralNetwork.feedForward(inputLayer.getNodes(), is);
			String output = "";
			for (Node n : outputLayer.getNodes()) output += n.getActivation() + " ";
			System.out.println(output);
		}
		System.out.println("E: " + FFNeuralNetwork.stdError(inputLayer.getNodes(), outputLayer.getNodes(), data));
		
	}

	/**
	 * test to see if neural net can represent bayesian transitions
	 * @return
	 */
	private static boolean bayesian() {
		System.out.println("BAYESIAN");
		
//		FFNeuralNetwork ffnn = new FFNeuralNetwork(ActivationFunction.SIGMOID0p5,5,5);
		ModelLearner modeler = new ModelLearnerHeavy(100, new int[] {}, new int[] {5},
				new int[] {}, ActivationFunction.SIGMOID0p5, 10);
		
		Collection<DataPoint> data = new ArrayList<DataPoint>();
		data.add(new DataPoint(new double[] {0,0,1,0,0}, new double[] {0,0,0,1,0})); // move right
		data.add(new DataPoint(new double[] {0,0,1,0,0}, new double[] {0,1,0,0,0})); // move left
		data.add(new DataPoint(new double[] {0,1,0,0,0}, new double[] {1,0,0,0,0})); // move left again
		data.add(new DataPoint(new double[] {0,0,0,1,0}, new double[] {0,0,1,0,0})); // move back to center
		data.add(new DataPoint(new double[] {0,0,0,1,0}, new double[] {0,0,0,0,0})); // disappear
		
//		ControlPanel.learnFromBackPropagation(ffnn.getInputNodes(), ffnn.getOutputNodes(), data,
//				10000, 1,1,0,0,0,0);
		for (DataPoint dp : data) {
			modeler.observePreState(dp.getInput());
			modeler.observePostState(dp.getOutput());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(1.5,0.5,0, false, 1000);
		modeler.getTransitionsModule().getNeuralNetwork().report(data);
		
		double[] foresight = Foresight.montecarlo(modeler, new double[] {0,0,1,0,0}, null, 1, 10000, 10, 0.1);
		for (double d : foresight) System.out.print(d + "	");
		System.out.println(near(foresight[0],0) && near(foresight[1],0.5) && near(foresight[2],0)
				&& near(foresight[3],0.5) && near(foresight[4],0)
				? "montecarlo 1 ok" : "montecarlo 1 sucks");
		foresight = Foresight.montecarlo(modeler, new double[] {0,0,1,0,0}, null, 2, 10000, 10, 0.1);
		for (double d : foresight) System.out.print(d + "	");
		System.out.println(near(foresight[0],0.5) && near(foresight[1],0) && near(foresight[2],0.25)
				&& near(foresight[3],0) && near(foresight[4],0)
				? "montecarlo 2 ok" : "montecarlo 2 sucks");

		return false;
	}
	
	private static boolean near(double d1, double d2) {
		return d1 < d2 + .1 && d1 > d2 - .1;
	}
	

	private static void envtranslator() {
		final EnvTranslator bucketer = EnvTranslator.rbfEnvTranslator(
				new double[] {0, 0, -2.5}, new double[] {1.0, 5.0, 2.5}, new int[] {5, 5, 3}, 0.8);
		double[] testVals = {.33, 5.0, -0.2};
		double[] ins = bucketer.toNN(testVals);
		double[] back = bucketer.fromNN(ins);
		String s1 = "", s2 = "";
		for (double d : ins) s1 += d + "	";
		for (double d : back) s2 += d + "	";
		System.out.println("INS	" + s1);
		System.out.println("BACK	" + s2);
	}
	
	private static void transitionApproximator() {
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {5}, 
				new int[] {5}, new int[] {5}, ActivationFunction.SIGMOID0p5, 50);
		double[][] inputSamples = {{0,0},{0,1},{1,0},{1,1}};
		double[][] outputSamples = {{0},{1},{1},{0}};
		Collection<DataPoint> data = DataPoint.createData(inputSamples, outputSamples);

		for (DataPoint dp : data) {
			double[] inputs = dp.getInput();
			modeler.observePreState(inputs[0]);
			modeler.observeAction(inputs[1]);
			modeler.observePostState(dp.getOutput());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(2, 0.5, 0, false, 1000);
		modeler.getModelVTA().getNeuralNetwork().report(data);
	}

	
	private static void fakePole() {
		//TODO pls try to get this working with more buckets
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(100, new int[] {30},
				new int[] {30}, new int[] {30}, ActivationFunction.SIGMOID0p5, 500);

		final boolean NN_FORM = false;
		double[] mins = Pole.stateMins;
		double[] maxes = Pole.stateMaxes;
		EnvTranslator stateTranslator = EnvTranslator.rbfEnvTranslator(mins, maxes, new int[] {12,12}, .5);
		EnvTranslator actTranslator = Pole.actionTranslator;
		List<double[]> actions = Pole.actionChoices;
		actions.add(new double[] {1,0});
//		actions.add(new double[] {0,0});
		actions.add(new double[] {0,1});
		final Collection<double[]> tmpCorrel = new ArrayList<double[]>();
		for (int t = 0; t < 2000; t++) {
			double[] preState = new double[mins.length];
			for (int i = 0; i < preState.length; i++) {
				final double spread = (maxes[i] - mins[i]) / 10;
				preState[i] = RandomUtils.randBetween(mins[i] - spread, maxes[i] + spread);	
			}
			double[] inNN = stateTranslator.toNN(preState);
			double[] action = RandomUtils.randomOf(actions);
			modeler.observePreState(inNN);
			modeler.observeAction(action);
			
			double[] postState = new double[mins.length];
			double act = Math.random() < 0.99 ? action[0] - action[1] : (2*Math.round(Math.random())-1);
			double r = act/100;
			for (int i = 0; i < postState.length; i++) {
				postState[i] = preState[i] * Math.exp(Math.signum(preState[i]) * (i == 0 ? r : -r));
			}
			tmpCorrel.add(new double[] {act, postState[0] - preState[0]});
			modeler.observePostState(stateTranslator.toNN(postState));
			modeler.saveMemory();
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 1000, 0.0003); // IT SEEMS CONFIDENCE IS NOT RELIABLE INDICATOR
//		for (double[] dd : tmpCorrel) {
//			String s = ""; for (double d : dd) s += d + "	";
//			System.out.println(s);
//		}
		for (int i = 0; i < 10; i++) System.out.println("*********");
		System.out.println(modeler.getModelVTA().getConfidenceEstimate());
		modeler.testit(1000, mins, maxes, stateTranslator, actTranslator, actions, NN_FORM);
	}

	private static void modeling() {
		final double[][] inputs = new double[][] {{0,0},{0,0},{0,0}};
		final double[][] outputs1 = new double[][] {{0,0},{1,1},{0,0}};
		final double[][] outputs2 = new double[][] {{1,0},{0,1},{0,1}};
		final DataPoint[] dp1 = new DataPoint[] {DataPoint.create(inputs[0], outputs1[0]),
				DataPoint.create(inputs[1], outputs1[1]), DataPoint.create(inputs[2], outputs1[2])};
		final DataPoint[] dp2 = new DataPoint[] {DataPoint.create(inputs[0], outputs2[0]),
				DataPoint.create(inputs[1], outputs2[1]), DataPoint.create(inputs[2], outputs2[2])};
		final DataPoint[] dps = Utils.append(dp1, dp2);
		final int trainingEpochs = 100;
		
		// scenario 1
		ModelLearnerHeavy modeler1 = learnModelFromData(dp1, trainingEpochs);
		// scenario 2
		ModelLearnerHeavy modeler2 = learnModelFromData(dp2, trainingEpochs);
		
//		printForData(modeler1, modeler1.getModelTFA(), dps);
//		System.out.println();
//		printForData(modeler2, modeler2.getModelTFA(), dps);
		
		// FAMILIARITY:
		TransitionMemory tm = new TransitionMemory(new double[] {0,0}, new double[] {}, new double[] {.5,.4});
		int jointRounds = 10;
//		double[] newFamiliar1 = modeler1.upFamiliarity(tm, jointRounds, jointRounds, 1.5, .5, 0, 0.1);
//		double[] newFamiliar2 = modeler2.upFamiliarity(tm, jointRounds, jointRounds, 1.5, .5, 0, 0.1);
//		System.out.println(newFamiliar1[0] + "," + newFamiliar1[1] + "	vs	" + newFamiliar2[0] + "," + newFamiliar2[1]);
	
		for (int i = 0; i < 8; i++) {
			double[] newJoint1 = modeler1.upJointOutput(tm, jointRounds);
			double[] newJoint2 = modeler2.upJointOutput(tm, jointRounds);
			System.out.println(Math.round(newJoint1[0]) + "," + Math.round(newJoint1[1])
					+ "	vs	" + Math.round(newJoint2[0]) + "," + Math.round(newJoint2[1]));
		}
		System.out.println();
	}
	
	private static ModelLearnerHeavy learnModelFromData(DataPoint[] data, int epochs) {
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500,
				new int[] {5}, new int[] {5}, new int[] {}, ActivationFunction.SIGMOID0p5, 50);
		for (DataPoint dp : data) {
			double[] inputs = dp.getInput();
			modeler.observePreState(inputs[0]);
			modeler.observeAction(inputs[1]);
			modeler.observePostState(dp.getOutput());
			modeler.saveMemory();
		}
		// TODO (not just testin) familiarity learning should run until error < e (.5?) on experience replay
		modeler.learnFromMemory(1.5, 0.5, 0, false, epochs, 0.01);
		return modeler;
	}
	
	public static void testModelerModule(ModelLearnerHeavy ml, ModelerModule mm, TestInputGenerator tig, int times) {
		for (int t = 0; t < times; t++) printForData(ml, mm, DataPoint.create(tig.generateTestInput(), null));
	}
	
	public static void printForData(ModelLearnerHeavy ml, ModelerModule mm, DataPoint... data) {
		for (DataPoint dp : data) {
			ml.observePreState(dp.getInput());
			if (dp.getOutput() != null) ml.observePostState(dp.getOutput());
			ml.feedForward();
			Collection<? extends Node> inputs = mm.getNeuralNetwork().getInputNodes();
			Collection<? extends Node> outputs = mm.getNeuralNetwork().getOutputNodes();
			String s = "I	";
			for (Node n : inputs) s += n.getActivation() + "	";
			s += "O	";
			for (Node n : outputs) s += n.getActivation() + "	";
			System.out.println(s);
		}
	}
	
	public static interface TestInputGenerator {
		public double[] generateTestInput();
	}
	public static TestInputGenerator createTestInputGenerator(double[]... samples) {
		final List<double[]> testSamples = new ArrayList<double[]>();
		for (double[] s : samples) testSamples.add(s);
		return new TestInputGenerator() {
			@Override
			public double[] generateTestInput() {
				return RandomUtils.randomOf(testSamples);
			}
		};
	}
}
