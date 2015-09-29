package transfertests;

import java.util.ArrayList;
import java.util.Collection;

import ann.BiasNode;
import ann.testing.FlierCatcher;
import ann.testing.GridExploreGame;
import modeler.ModelLearnerHeavy;
import modularization.ModuleDisplayer;

/**
 * IS THIS EVEN WORTH IT? rules are so simple it's just as easy to train I/O from scratch
 * than to try to "reuse"
 */

public class T4ReuseSource {
	final static int size = 5;
	final static int trainTurns = 500;
	final static int testTurns = 100;
	final static int repaintMs = 50;
	final static int learnIterations = 30;

	public static void main(String[] args) {
	}
	
	public static FlierCatcher setupReuse(FlierCatcher source) {

		FlierCatcher fcFull = new FlierCatcher(size);
		ModelLearnerHeavy soloModel = TransferTestUtils.loadOrCreate("T4SOLO", fcFull, trainTurns);
		return null;
	}

	public static FlierCatcher setupHalfGame() {
		FlierCatcher fcSolo = new FlierCatcher(size);
		fcSolo.setSpawnFreq(0);
		ModelLearnerHeavy soloModel = TransferTestUtils.loadOrCreate("T4SOLO", fcSolo, trainTurns);
		System.out.println("Learning Partial Game...");
		soloModel.learnFromMemory(0.5, 0.5, 0, false, learnIterations , 10000);
		fcSolo.modeler = soloModel;
		return fcSolo;
	}

	public static FlierCatcher setupScratchGame() {
		FlierCatcher fcScratchFliers = new FlierCatcher(size);
		ModelLearnerHeavy scratchFlierModel = TransferTestUtils.loadOrCreate("T4SCRATCH", fcScratchFliers, trainTurns);
		System.out.println("Learning Scratch Game...");
		scratchFlierModel.learnFromMemory(0.5, 0.5, 0, false, learnIterations , 10000);
		fcScratchFliers.modeler = scratchFlierModel;
		return fcScratchFliers;
	}
	
	private static void play(FlierCatcher game) {
		int epochs = 100;
		int numSteps = 3;
		int numRuns = 5;
		int joints = 1;
		double skewFactor = 0.1;
		double discRate = 0.2;
		FlierCatcher.play(game.modeler, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
	}
}
