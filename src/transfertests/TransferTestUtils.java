package transfertests;

import java.util.Collection;

import ann.ActivationFunction;
import ann.Utils;
import ann.testing.FlierCatcher;
import ann.testing.GridExploreGame;
import modeler.ModelLearnerHeavy;

public class TransferTestUtils {
	public static ModelLearnerHeavy loadOrCreate(String name, FlierCatcher game, int turns) {
		return loadOrCreate(name, game, turns, new int[] {game.cols*game.rows*2}, new int[] {game.cols*game.rows*3});
	}
	public static ModelLearnerHeavy loadOrCreate(String name, FlierCatcher game, int turns,
			int[] pHL, int[] cHL) {
		game.modeler = Utils.loadModelerFromFile(name, turns);
		int t = 1;
		if (game.modeler == null) {
			game.modeler = new ModelLearnerHeavy(500, pHL, null, cHL, ActivationFunction.SIGMOID0p5, turns);
			t = turns;
		}
		FlierCatcher.trainModeler(game.modeler, t, game, 0, game.actionChoices, GridExploreGame.actionTranslator);
		return game.modeler;
	}
	
	public static double tTest(Collection<Double> ds) {
		double m = mean(ds);
		return m / stdev(ds, m) * Math.sqrt(ds.size());
	}
	public static double mean(Collection<Double> ds) {
		double result = 0;
		for (double d : ds) result += d;
		return result / ds.size();
	}
	public static double stdev(Collection<Double> ds, double mean) {
		double result = 0;
		for (double d : ds) result += Math.pow(d - mean, 2);
		return Math.sqrt(result / ds.size());
	}
	
	public static double compareTwoModelers(FlierCatcher game, int epochs, ModelLearnerHeavy modeler1, ModelLearnerHeavy modeler2,
			boolean fastForward) {
		FlierCatcher.repaintMs = fastForward ? -1 : 50;
		int numSteps = 3;
		int numRuns = 5;
		int joints = 1;
		double skewFactor = 0.1;
		double discRate = 0.2;
		FlierCatcher.play(modeler1, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		final double wr1 = game.getWinRate();
		game.clearWinRate();
		FlierCatcher.play(modeler2, game, epochs, numSteps, numRuns, joints, skewFactor, discRate);
		final double wr2 = game.getWinRate();
		System.out.println(wr1 + "	vs	" + wr2);
		return wr2 - wr1;
	}
}
