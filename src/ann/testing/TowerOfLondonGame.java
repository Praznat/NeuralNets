package ann.testing;

import java.awt.*;
import java.util.*;

public class TowerOfLondonGame extends GridGame {
	

	enum Chip {RED, GREEN, BLUE};

	Point redPt, greenPt, bluePt;
	Collection<Point> points = new ArrayList<Point>();
	
	public TowerOfLondonGame() {
		super(3, 3);
		points.add(redPt);
		points.add(greenPt);
		points.add(bluePt);
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	@Override
	public double[] getState() {
		double[] result = new double[6];
		int i = 0;
		for (Point pt : points) {
			result[i++] = pt.getX();
			result[i++] = pt.getY();
		}
		return result;
	}

	@Override
	public void oneTurn() {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected void paintGrid(Graphics g) {
		// TODO Auto-generated method stub
		
	}

}
