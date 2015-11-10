package ann.testing;

import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;

import ann.ActivationFunction;
import ann.Utils;
import ann.testing.display.GridGameDisplay;
import modeler.EnvTranslator;
import modeler.ModelLearnerHeavy;
import reasoner.Planner;
import reasoner.RewardFunction;

public abstract class GridGame implements IGridGame {
	public ModelLearnerHeavy modeler;
	protected GridGameDisplay display;

	public final int rows;
	public final int cols;
	public final List<double[]> actionChoices = new ArrayList<double[]>();
	
	double[] chosenAction;
	RewardFunction rewardFn;
	boolean isPaused;
	final Thread thread;
	public GridGame(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		this.thread = Thread.currentThread();
	}
	
	public int getRows() {return rows;};
	public int getCols() {return cols;};
	public List<double[]> getActionChoices() {return actionChoices;}
	
	protected static int getGridLoc(int col, int row, int[][] grid) {
		int cols = grid.length;
		int rows = grid[0].length;
		return grid[(col + cols) % cols][(row + rows) % rows];
	}
	public static double sum(double[][] grid) {
		double sum = 0;
		for (int c = 0; c < grid.length; c++) {
			for (int r = 0; r < grid[c].length; r++) {
				sum += grid[c][r];
			}
		}
		return sum;
	}
	public static double sum(double[] state) {
		double sum = 0;
		for (double d : state) sum += d;
		return sum;
	}
	public static double pctCorrect(int[][] grid1, double[][] grid2, boolean justOnes) {
		double sum = 0;
		int n = 0;
		for (int c = 0; c < grid1.length; c++) {
			for (int r = 0; r < grid1[c].length; r++) {
				int o1 = grid1[c][r];
				if (justOnes && o1 < 1) continue;
				double err = o1 - grid2[c][r];
				sum += Math.abs(err);
				n++;
			}
		}
		return n == 0 ? Double.NaN : Utils.round(1 - sum/n, 4) * 100;
	}
	
	public abstract double[] getState();
	
	public void convertFromState(double[] state) {
		System.out.println("Error: Need to implement (override) GridGame.convertFromState");
	}

	public abstract void oneTurn();
	
	public void setupForTurn() {}

	public RewardFunction getRewFun() {
		return rewardFn;
	}

	RewardFunction getRewardFunction() {
		return rewardFn;
	}
	public void setRewardFunction(RewardFunction rewardFunction) {
		 this.rewardFn = rewardFunction;
	}
	
	public void repaint() {
		if (display != null) display.frame.repaint();
	}
	
	public void setupGameDisplay() {
		display = new GridGameDisplay();
		display.gridPanel.setGame(this);
	}

	public void setupControlPanel() {
		display.controlPanel.setGame(this);
	}
	
	public int getGUnit() {
		return display.gridPanel.getgUnit();
	}
	
	public static void trainModeler(ModelLearnerHeavy modeler, int turns, GridGame game,
			int repaintMs, List<double[]> actionChoices, EnvTranslator actionTranslator) {
		Planner chimp = Planner.createRandomChimp();
		boolean repaint = repaintMs > 0;
		for (int t = 0; t < turns; t++) {
			long startMs = System.currentTimeMillis();
			double[] actionNN = chimp.getOptimalAction(game.getState(), actionChoices, 0, 0.1);
			game.chosenAction = actionTranslator.fromNN(actionNN);
			game.setupForTurn();
			modeler.observePreState(game.getState());
			modeler.observeAction(actionTranslator.toNN(game.chosenAction));
			game.oneTurn();
			game.repaint();
			modeler.observePostState(game.getState());
			modeler.saveMemory();
			if (repaint) try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, repaintMs - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	protected static ModelLearnerHeavy trainedModeler(int numVars, int numStates, GridGame game, double sampleSizeMultiplier,
			int repaintMs, List<double[]> actionChoices, EnvTranslator actionTranslator, int[] jdmHL) {
		int turns = (int) (numStates * sampleSizeMultiplier);
		System.out.println("Observing " + turns + " samples");
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {numVars*2},
				null, jdmHL, ActivationFunction.SIGMOID0p5, turns);
		trainModeler(modeler, turns, game, repaintMs, actionChoices, actionTranslator);
		return modeler;
	}
	
	public abstract void paintGrid(Graphics g);
	
	public static void print(double[][] grid) {
		for (int r = 0; r < grid[0].length; r++) {
			String s = "";
			for (int c = 0; c < grid.length; c++) {
				s += Utils.round(grid[c][r], 2) + "	";
			}
			System.out.println(s);
		}
		System.out.println("***********");
	}

}

