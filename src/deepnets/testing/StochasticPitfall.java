package deepnets.testing;

import java.awt.*;
import java.util.*;
import java.util.List;

import modeler.ModelLearnerHeavy;

public class StochasticPitfall extends GridTagGame {

	protected int hits;
	protected double[] oldAction;
	
	protected ModelLearnerHeavy modeler;
	protected Point lastPlayerPos;
	protected static final int HEIGHT = 4;
	protected final PlayerRule GUY_MOVEMENT = new PlayerRule() {
		@Override
		Point posTo(Point from, double[] action) {
			boolean onFloor = from.y == floorRow();
			boolean atCeiling = from.y == highestRow();
			int newX = onFloor ? from.x + (int) (action[2] - action[0])
					: 2 * from.x - lastPlayerPos.x;
			int newY = onFloor ? from.y - (int) action[1]			// jump from floor
					: atCeiling ? highestRow() + 1		// fall from ceiling
							: 2 * from.y - lastPlayerPos.y;
			if (onFloor) oldAction = action;
			else if (atCeiling) oldAction = new double[] {oldAction[0], 0, oldAction[2]};
			return new Point(newX, newY);
		}
	};
	protected static final List<double[]> ACTION_CHOICES = new ArrayList<double[]>();
	static {
		ACTION_CHOICES.add(new double[] {0,0,0}); //idle
		ACTION_CHOICES.add(new double[] {1,0,0}); //left
		ACTION_CHOICES.add(new double[] {0,0,1}); //right
		ACTION_CHOICES.add(new double[] {0,1,0}); //jump up
		ACTION_CHOICES.add(new double[] {1,1,0}); //jump left
		ACTION_CHOICES.add(new double[] {0,1,1}); //jump right
	}
	protected int epochsPast;
	
	public StochasticPitfall(int cols) {
		super(HEIGHT, cols);
		playerPos = new Point(0, floorRow());
		lastPlayerPos = playerPos.getLocation();
		setPlayerPos(0, floorRow());
		playerRule = GUY_MOVEMENT;
		gridPanel.setGame(this);
	}
	
	public void oneTurn() {
		Point tmp = playerPos.getLocation();
		playerRule.move(chosenAction);
		lastPlayerPos = tmp;
		epochsPast++;
		if (opponentGrid[playerPos.x][playerPos.y] > 0.5) hits++;
		frame.repaint();
	}
	
	protected void resetPoints() {
		epochsPast = 0;
		hits = 0;
	}
	
	protected double getScore() {
		return -((double)hits)/epochsPast;
	}
	
	@Override
	protected void setPlayerPos(int col, int row) {
		playerGrid[playerPos.x][playerPos.y-1] = 0;
		setPos(col, row, playerPos, playerGrid);
		playerGrid[playerPos.x][playerPos.y-1] = 1;
	}
	protected void setPos(int col, int row, Point pos, int[][] grid) {
		grid[pos.x][pos.y] = 0;
		pos.setLocation(Math.min(Math.max(col, 0), cols-1),
				Math.min(Math.max(row, 0), rows-1));
		grid[pos.x][pos.y] = 1;
	}
	protected int floorRow() {
		return rows-1;
	}
	protected int highestRow() {
		return 1;
	}
	protected boolean isPlayerJumping() {
		return playerPos.y < floorRow();
	}

	@Override
	protected void paintGrid(Graphics g) {
		final int gSub = 500 / cols;
		final int thinness = gSub / 16;
		int[][] pGrid = this.opponentGrid;
		g.setColor(Color.GREEN);
		g.fillRect(0, 0, cols*gSub, rows*gSub);
		for (int c = 0; c < pGrid.length; c++) {
			int[] col = pGrid[c];
			for (int r = 0; r < col.length; r++) {
				if (playerGrid[c][r] > 0.5) {
					if (r > 0 && playerGrid[c][r-1] > 0.5) { // body
						g.setColor(Color.BLUE);
						g.fillRect(c*gSub + thinness, r*gSub, gSub - 2*thinness, gSub);
					} else {
						g.setColor(Color.YELLOW); // head
						g.fillOval(c*gSub, r*gSub, gSub, gSub);
						g.setColor(Color.BLACK);
						g.fillOval((int) ((c+0.6)*gSub), (int) ((r+0.3)*gSub), gSub/8, gSub/8);
						g.fillOval((int) ((c+0.85)*gSub), (int) ((r+0.25)*gSub), gSub/9, gSub/9);
						g.fillOval((int) ((c+0.8)*gSub), (int) ((r+0.5)*gSub), gSub/5, gSub/8);
					}
				}
				if (col[r] > 0.5) paintObstacle(g, c, r, gSub);
			}
		}
		morePainting(g, gSub);
	}
	
	protected void paintObstacle(Graphics g, int c, int col, int gSub) {}
	
	protected void morePainting(Graphics g, int gSub) {}

}
