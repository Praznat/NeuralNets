package transfertests;

import java.util.ArrayList;
import java.util.Collection;

import ann.BiasNode;
import ann.testing.FlierCatcher;
import ann.testing.GridExploreGame;
import modeler.ModelLearnerHeavy;
import modularization.ModuleDisplayer;

/**
 * when testTurns/learnIterations are low, prior network does significantly worse than scratch
 * maybe this is because the weights to the Flier cells are too weak and dont learn quickly enough?
 */

public class T2SourceAsPrior {
	final static int size = 5;
	final static int trainTurns = 500;
	final static int testTurns = 100;
	final static int repaintMs = 50;
	final static int learnIterations = 30;

	public static void main(String[] args) {
		multiTest(1);
	}
	
	public static void multiTest(int n) {
		Collection<Double> testScores = new ArrayList<Double>();
		for (int i = 0; i < n; i++) {
			double s = testTrainFromPartialSource();
			testScores.add(s);
		}
		for (double s : testScores) System.out.println(s);
		System.out.println("T = " + TransferTestUtils.tTest(testScores));
	}

	public static double testTrainFromPartialSource() {
		int epochs = 100;
		int numSteps = 3;
		int numRuns = 5;
		int joints = 1;
		double skewFactor = 0.1;
		double discRate = 0.2;
		
		FlierCatcher fcSolo = new FlierCatcher(size);
		fcSolo.setSpawnFreq(0);
		int[] gridSpec = new int[] {fcSolo.rows, fcSolo.cols};
		ModelLearnerHeavy priorModel = TransferTestUtils.loadOrCreate("T2SOLO", fcSolo, trainTurns);
		System.out.println("Learning Partial Game...");
		priorModel.learnFromMemory(0.5, 0.5, 0, false, learnIterations , 10000);
//		modularizeAndDisplay(priorModel, gridSpec);
		
		FlierCatcher fcScratchFliers = new FlierCatcher(size);
		ModelLearnerHeavy scratchFlierModel = TransferTestUtils.loadOrCreate("T2SCRATCH", fcScratchFliers, testTurns);
		scratchFlierModel.learnFromMemory(0.5, 0.5, 0, false, learnIterations , 10000);
		System.out.println("Learning Scratch Game...");
		
		FlierCatcher fcPrioredFliers = new FlierCatcher(size);
		FlierCatcher.trainModeler(priorModel, testTurns, fcPrioredFliers, 0, fcPrioredFliers.actionChoices, GridExploreGame.actionTranslator);
		priorModel.learnFromMemory(0.5, 0.5, 0, false, learnIterations , 10000);
		System.out.println("Retraining Priored Game...");
		
		FlierCatcher.repaintMs = 50;
		FlierCatcher.play(scratchFlierModel, fcScratchFliers, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		FlierCatcher.play(priorModel, fcPrioredFliers, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		
		System.out.println(fcScratchFliers.getWinRate() + "	vs	" + fcPrioredFliers.getWinRate());
		System.out.println(fcScratchFliers.getWinRate() < fcPrioredFliers.getWinRate() ? "Success" : "Failure");

		modularizeAndDisplay(priorModel, gridSpec);
		modularizeAndDisplay(scratchFlierModel, gridSpec);
		
		BiasNode.clearConnections();
		
		return fcPrioredFliers.getWinRate() - fcScratchFliers.getWinRate();
	}
	
	private static void modularizeAndDisplay(ModelLearnerHeavy model, int[] gridSpec) {
		T3IsolateIndependentRegions.modularize(model, 0.9, 1);
		ModuleDisplayer moduleDisplayer = new ModuleDisplayer(model, 2, gridSpec, gridSpec, new int[] {4, 1});
	}
}
