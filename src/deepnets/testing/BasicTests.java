package deepnets.testing;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import modeler.*;

import reasoner.*;
import utils.RandomUtils;

import deepnets.*;

public class BasicTests {

	public static void main(String[] args) {
		bayesian();
		xor();
		modelingContinuous();
		transitionApproximator();
		testKWIK();
		modeling();
		fakePole();
		envtranslator();
	}

	private static void xor() {
		System.out.println("RAW XOR");
		Layer<Node> inputLayer = Layer.createInputLayer(2, Node.BASIC_NODE_FACTORY);
		Layer<Node> hiddenLayer = Layer.createHiddenFromInputLayer(inputLayer, 4,
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
		
		ControlPanel.learnFromBackPropagation(inputLayer.getNodes(), outputLayer.getNodes(), data, 1000,
				0.9, 0.8, 0, 0.7, 0.3, 0);

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
		
		double[] foresight = Foresight.montecarlo(modeler, new double[] {0,0,1,0,0}, null, null, 1, 10000, 10, 0.1);
		for (double d : foresight) System.out.print(d + "	");
		System.out.println(near(foresight[0],0) && near(foresight[1],0.5) && near(foresight[2],0)
				&& near(foresight[3],0.5) && near(foresight[4],0)
				? "montecarlo 1 ok" : "montecarlo 1 sucks");
		foresight = Foresight.montecarlo(modeler, new double[] {0,0,1,0,0}, null, null, 2, 10000, 10, 0.1);
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
		System.out.println("modeling xor");
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
	
	// TODO test case where it gets so good at predicting that no more error propagates
	// that results in no additions to cumSqrChg
	// you need to penalize these states by adding uniformly to cumSqrChg on active inputs when output error is low
	private static void testKWIK() {
		int turns = 100;
		double[] winRate = {0,0,0,0};
		for (int i = 0; i < turns; i++) {
			double[] wins = testKWIKOnce();
			for (int j = 0; j < wins.length; j++) winRate[j] += wins[j];
		}
		for (int j = 0; j < winRate.length; j++) winRate[j] /= turns;
		System.out.println("KWIK scores");
		System.out.println(Utils.stringArray(winRate, 4));
	}
	private static double[] testKWIKOnce() {
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {15}, 
				new int[] {15}, new int[] {15}, ActivationFunction.SIGMOID0p5, 50);
		double[][] inputSamples = {{0,0,1,0},{0,1,0,0},{1,0,0,0}};
		double[][] outputSamples = {{0,0,0,1},{0,0,1,0},{0,1,0,0}};
		Collection<DataPoint> data = DataPoint.createData(inputSamples, outputSamples);

		int iterations = 50;
		for (int i = 0; i < iterations; i++) {
			for (DataPoint dp : data) {
				modeler.observePreState(dp.getInput());
				modeler.observePostState(dp.getOutput());
				modeler.learnOnline(1.5, 0.5, 0);
			}
		}
		data.add(new DataPoint(new double[] {0,0,0,1}, new double[] {1,0,0,0}));
		FFNeuralNetwork vta = modeler.getModelVTA().getNeuralNetwork();
		FFNeuralNetwork jdm = modeler.getModelJDM().getNeuralNetwork();
		Node familiode = jdm.getOutputNodes().get(jdm.getOutputNodes().size()-1);
//		vta.report(data);
		double p0 = 9999, p1 = 9999, p2 = 9999, p3 = 9999;
		double[] bests = {9999,9999,9999,9999};
		double[] wins = {0, 0, 0, 0};
		for (DataPoint dp : data) {
			FFNeuralNetwork.feedForward(vta.getInputNodes(), dp.getInput());
			FFNeuralNetwork.feedForward(jdm.getInputNodes(), dp.getInput());
			double[] outputs = Utils.getActivations(vta.getOutputNodes());
			p0 = Foresight.estimateCertainty(outputs);
			p1 = familiode.getActivation();
			p2 = Foresight.estimateWeightCertainty(vta.getInputNodes(), false);
			p3 = Foresight.estimateWeightCertainty(vta.getInputNodes(), true);
			if (p0 < bests[0]) bests[0] = p0;
			if (p1 < bests[1]) bests[1] = p1;
			if (p2 < bests[2]) bests[2] = p2;
			if (p3 < bests[3]) bests[3] = p3;
//			System.out.println(Utils.stringArray(dp.getInput(), 0) + "	" + Utils.stringArray(outputs, 4)
//					+ "	:" + p0 + "	:" + p1 + "	:" + p2 + "	:" + p3);
		}
		if (p0 == bests[0]) wins[0]++;
		if (p1 == bests[1]) wins[1]++;
		if (p2 == bests[2]) wins[2]++;
		if (p3 == bests[3]) wins[3]++;
		BiasNode.clearConnections();
		return wins;
	}
	
	private static void fakePole() {
		//TODO pls try to get this working with more buckets
		int turns = 1000;
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(100, new int[] {30},
				new int[] {30}, new int[] {30}, ActivationFunction.SIGMOID0p5, turns);

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
		for (int t = 0; t < turns; t++) {
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
			} // act0 moves state[0] up and state[1] down, act1 moves state[0] down and state[1] up
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
//		modeler.testit(1000, mins, maxes, stateTranslator, actTranslator, actions, NN_FORM);
		
		for (int t = 0; t < 500; t++) {
			final double[] state = new double[mins.length];
			for (int i = 0; i < state.length; i++) state[i] = RandomUtils.randBetween(mins[i], maxes[i]);
			String s = "";
			for (double d : state) s += d + "	";
			for (double[] act : actions) {
				double[] foresight = Foresight.montecarlo(modeler, stateTranslator.toNN(state), act, 1, 100, 30);
				double[] postV = stateTranslator.fromNN(foresight);
				s += "act:" + actTranslator.fromNN(act) + ":	";
				for (double d : postV) s += Utils.round(d * 100, 2) + "	";
			}
			System.out.println(s);
		}
	}

	private static void modelingContinuous() {
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {5}, 
				new int[] {5}, new int[] {5}, ActivationFunction.SIGMOID0p5, 50);
		double[][] inputSamples = {{0.55,0.45,1},{0.35,0.65,1},{0.25,0.75,-1},{0.85,0.15,-1}};
		double[][] outputSamples = {{0.75, 0.25},{0.55, 0.45},{0.05, 0.95},{0.65, 0.35}};
		Collection<DataPoint> data = DataPoint.createData(inputSamples, outputSamples);

		for (DataPoint dp : data) {
			double[] inputs = dp.getInput();
			modeler.observePreState(new double[] {inputs[0], inputs[1]});
			modeler.observeAction(inputs[2]);
			modeler.observePostState(dp.getOutput());
			modeler.saveMemory();
		}
		modeler.learnFromMemory(2, 0.5, 0, false, 1000);
		modeler.getModelVTA().getNeuralNetwork().report(data);
		for (int i = 0; i < inputSamples.length; i++) {
			double[] foresight = Foresight.montecarlo(modeler, inputSamples[i], null, null, 1, 10000, 10, 0.1);
			for (double d : foresight) System.out.print(d + "	");
			System.out.println();
		}
	}
	
	private static void modeling() {
		final double[][] inputs = new double[][] {{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1},{1,1}};//{{0,0},{0,0},{0,0}};
		final double[][] outputs1 = new double[][] {{0,0},{1,1},{0,0},{1,0},{0,0},{1,1},{0,0},{0,1},{0,0},{1,1},{0,0}};
		final double[][] outputs2 = new double[][] {{1,0},{0,1},{0,1},{0,0},{1,0},{0,1},{0,1},{1,1},{1,0},{0,1},{0,1}};
		final DataPoint[] dp1 = new DataPoint[outputs1.length];
		final DataPoint[] dp2 = new DataPoint[outputs2.length];
		for (int i = 0; i < inputs.length; i++) {
			dp1[i] = DataPoint.create(inputs[i], outputs1[i]);
			dp2[i] = DataPoint.create(inputs[i], outputs2[i]);
		}
//		{DataPoint.create(inputs[0], outputs1[0]),
//				DataPoint.create(inputs[1], outputs1[1]), DataPoint.create(inputs[2], outputs1[2])};
//		final DataPoint[] dp2 = new DataPoint[] {DataPoint.create(inputs[0], outputs2[0]),
//				DataPoint.create(inputs[1], outputs2[1]), DataPoint.create(inputs[2], outputs2[2])};
		final DataPoint[] dps = Utils.append(dp1, dp2);
		final int trainingEpochs = 1000;
		
		// scenario 1
		ModelLearnerHeavy modeler1 = learnModelFromData(dp1, trainingEpochs);
		// scenario 2
		ModelLearnerHeavy modeler2 = learnModelFromData(dp2, trainingEpochs);
		int numRuns = 1000;
		int jointAdjustments = 10;
		double skewFactor = 0;
		double cutoffProb = 0.1;
		DecisionProcess decisioner1 = new DecisionProcess(modeler1, null, 1, numRuns,
				jointAdjustments, skewFactor, 0, cutoffProb);
		DecisionProcess decisioner2 = new DecisionProcess(modeler2, null, 1, numRuns,
				jointAdjustments, skewFactor, 0, cutoffProb);
		Map<DiscreteState,AtomicInteger> map1 =
				decisioner1.getImmediateStateGraphForActionGibbs(inputs[0], new double[] {});
		Map<DiscreteState,AtomicInteger> map2 =
				decisioner2.getImmediateStateGraphForActionGibbs(inputs[0], new double[] {});
		System.out.println(map1);
		System.out.println("#1 Should be approx:	{=545, 0.1.=273, other=99}");
		System.out.println(map2);
		System.out.println("#2 Should be approx:	{1.=545, 0.=273, other=99}");
		
//		printForData(modeler1, modeler1.getModelTFA(), dps);
//		System.out.println();
//		printForData(modeler2, modeler2.getModelTFA(), dps);
		
		// FAMILIARITY:
		TransitionMemory tm = new TransitionMemory(new double[] {0,0}, new double[] {}, new double[] {.5,.4});
		int jointRounds = 10;
//		double[] newFamiliar1 = modeler1.upFamiliarity(tm, jointRounds, jointRounds, 1.5, .5, 0, 0.1);
//		double[] newFamiliar2 = modeler2.upFamiliarity(tm, jointRounds, jointRounds, 1.5, .5, 0, 0.1);
//		System.out.println(newFamiliar1[0] + "," + newFamiliar1[1] + "	vs	" + newFamiliar2[0] + "," + newFamiliar2[1]);
	
		double[] z1 = new double[4]; double[] z2 = new double[4];
		int n = 500;
		for (int i = 0; i < n; i++) {
			double[] newJoint1 = modeler1.upJointOutput(tm, jointRounds);
			double[] newJoint2 = modeler2.upJointOutput(tm, jointRounds);
//			System.out.println(Math.round(newJoint1[0]) + "," + Math.round(newJoint1[1])
//					+ "	vs	" + Math.round(newJoint2[0]) + "," + Math.round(newJoint2[1]));
			z1[(int)(Math.round(newJoint1[0])*2+Math.round(newJoint1[1]))]++;
			z2[(int)(Math.round(newJoint2[0])*2+Math.round(newJoint2[1]))]++;
		}
		System.out.println("	[0,0]	[0,1]	[1,0]	[1,1]");
		System.out.println("+CORR:	"+z1[0]/n+"	"+z1[1]/n+"	"+z1[2]/n+"	"+z1[3]/n);
		System.out.println("-CORR:	"+z2[0]/n+"	"+z2[1]/n+"	"+z2[2]/n+"	"+z2[3]/n);
		System.out.println("+CORR should be 1/3 1,1 and 2/3 0,0. -CORR should be 1/3 1,0 and 2/3 0,1");
	}
	
	private static ModelLearnerHeavy learnModelFromData(DataPoint[] data, int epochs) {
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500,
				new int[] {12}, new int[] {8}, new int[] {15}, ActivationFunction.SIGMOID0p5, 50);
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
