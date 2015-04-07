package deepnets.testing;

import java.awt.*;

import reasoner.Foresight;

import deepnets.ActivationFunction;

import modeler.*;

public class CellAutomataGame extends GridGame {
	
	static final CellUpdateRule GAME_OF_LIFE = new CellUpdateRule() {
		@Override
		public boolean turnOn(int r, int c, int[][] grid) {
			int numNeighbors = 0;
			for (int dx = -1; dx <= 1; dx ++) {
				for (int dy = -1; dy <= 1; dy ++) {
					numNeighbors += getGridLoc(c + dx, r + dy, grid);
				}
			}
			if (getGridLoc(c, r, grid) > 0) {
				numNeighbors--; // already counted self
				return numNeighbors == 3;
			} else {
				return numNeighbors == 2 || numNeighbors == 3;
			}
		}
	};

	final int[][] grid; // 1 or 0
	final boolean[][] nextState;
	final CellUpdateRule updateRule;
	int mode = 0;

	public CellAutomataGame(int rows, int cols, CellUpdateRule updateRule) {
		super(rows, cols);
		grid = new int[cols][rows];
		nextState = new boolean[cols][rows];
		this.updateRule = updateRule;
	}
	
	void initialize(double pOn) {
		for (int c = 0; c < cols; c++) for (int r = 0; r < rows; r++) {
			grid[c][r] = Math.random() < pOn ? 1 : 0;
		}
	}

	@Override
	public void oneTurn() {
		for (int c = 0; c < cols; c++) for (int r = 0; r < rows; r++) {
			nextState[c][r] = GAME_OF_LIFE.turnOn(r, c, grid);
		}
		for (int c = 0; c < cols; c++) for (int r = 0; r < rows; r++) {
			grid[c][r] = nextState[c][r] ? 1 : 0;
		}
	}
	
	public double[] getState() {
		double[] result = new double[rows * cols];
		for (int c = 0; c < cols; c++) {
			int[] col = grid[c];
			for (int r = 0; r < rows; r++) result[c + r*cols] = col[r];
		}
		return result;
	}
	public double[][] getGridFromState(double[] state) {
		double[][] result = new double[cols][rows];
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c][r] = state[c + r*cols];
		}
		return result;
	}
	public void setGrid(double[][] other) {
		for (int c = 0; c < cols; c++) {
			int[] col = grid[c];
			for (int r = 0; r < rows; r++) col[r] = (int) Math.round(other[c][r]);
		}
	}
	
	public static void main(String[] args) {
		test1();
	}
	static void test1() {
		int size = 4;
		int turns = size*size*500;
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {size * size * 2},
				new int[] {}, new int[] {}, ActivationFunction.SIGMOID0p5, turns);
		int tests = 10;
		long trainingTimeMs = 1000;
		double start = 0.1;
		double end = 0.6;
		for (int i = 0; i < tests; i++) {
			double pOn = start + (end - start)*(((double)i)/(tests-1));//Math.random();
			System.out.println("playing cell automata with " + pOn);
			observeGame(modeler, size, turns / tests, pOn, trainingTimeMs, true);
			System.out.println("finished cell automata");
		}
		modeler.learnFromMemory(1.5, 0.5, 0, false, 500, 30000);
		
		CellAutomataGame game = new CellAutomataGame(size, size, GAME_OF_LIFE);
		game.mode = 1;
		game.initialize(Math.random());

		simulateImaginarily(game, modeler, 10000, 500, true);
	}
	static void observeGame(ModelLearner modeler, int size, int epochs, double pOn, long lengthMs, boolean reInit) {
		CellAutomataGame game = new CellAutomataGame(size, size, GAME_OF_LIFE);
		gridPanel.setGame(game);
		game.initialize(pOn);
		long frameMs = lengthMs / epochs;
		for (int t = 0; t < epochs; t++) {
			if (reInit) game.initialize(pOn);
			long startMs = System.currentTimeMillis();
			if (modeler != null) modeler.observePreState(game.getState());
			game.oneTurn();
			if (modeler != null) {
				double[] postState = game.getState();
				modeler.observePostState(postState);
				modeler.saveMemory();
				if (sum(postState) < 3) game.initialize(pOn);
			}
			frame.repaint();
			try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, frameMs - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	static void simulateImaginarily(CellAutomataGame game, ModelLearner modeler, int epochs, long frameMs,
			boolean alsoLearn) {
		int numRuns = 5;
		int jointAdjustments = 0;
		gridPanel.setGame(game);
		frame.repaint();
		boolean validPre = false;
		for (int t = 0; t < epochs; t++) {
			double[] state = game.getState();
			if (sum(state) < 3) {
				game.initialize(Math.random());
				validPre = false;
				state = game.getState();
			}
			frame.repaint();
			if (alsoLearn) {
				if (validPre) {
					modeler.observePostState(state); // TODO NO!!! real state!
					modeler.saveMemory();
				}
				modeler.observePreState(state);
			}
			validPre = true;
			long startMs = System.currentTimeMillis();
			double[] guessState = Foresight.getBestPredictedNextState(modeler, state, numRuns, jointAdjustments);
			double[][] grid = game.getGridFromState(guessState);
			GridGame.print(grid);
			game.oneTurn(); // for comparing error
			System.out.println("% CORRECT:	" + GridGame.pctCorrect(game.grid, grid, true) + "	" + GridGame.pctCorrect(game.grid, grid, false));
			game.setGrid(grid);
			try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, frameMs - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		if (alsoLearn) modeler.learnFromMemory(1.5, 0.5, 0, false, 200, 30000);
	}

	@Override
	protected void paintGrid(Graphics g) {
		final int rHgt = gridPanel.gUnit / rows;
		final int cWid = gridPanel.gUnit / cols;
		g.setColor(this.mode == 0 ? Color.BLACK : Color.BLUE);
		int[][] pGrid = this.grid;
		for (int c = 0; c < pGrid.length; c++) {
			int[] col = pGrid[c];
			for (int r = 0; r < col.length; r++) {
				if (col[r] > 0) g.fillRect(c*cWid, r*rHgt, cWid, rHgt);
				else g.drawRect(c*cWid, r*rHgt, cWid, rHgt);
			}
		}
	}
}


interface CellUpdateRule {
	boolean turnOn(int r, int c, int[][] grid);
}