#include <iostream>
#include <queue>
#include <stack>
#include <vector>
#include <limits>
#include <sstream>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

double s;
double a;
double b;
double e;

//optimized	
/*double f(double x){
	double denominator = 1;
	double outerSum = 0;
	double innerSum = 0;
	for (int i = 1; i <= 100; i++){
		innerSum += pow(x + i, -3.1);
		denominator *= 1.2;
		outerSum += sin(x + innerSum) / denominator;
	}
	return outerSum;
}*/

//non-optimized
double f(double x){
	double outerSum = 0;
	for (int i = 1; i <= 100; i++){
		double innerSum = 0;
		for (int j = 1; j <= i; j++){
			innerSum += pow(x + j, -3.1);
		}
		outerSum += sin(x + innerSum) / pow(1.2, i);
	}
	return outerSum;
}

//test function
//double f(double x){
//	if (x < 40){
//		return 0;
//	}
//	if (x > 50){
//		return 0;
//	}
//	if (x < 45){
//		return 12 * (x - 40);
//	}
//	return 60 - 12 * (x - 45);
//}

struct Interval{
public:
	double a, b, fa, fb, max;

	Interval(double a0, double b0, double fa0, double fb0){
		a = a0;
		b = b0;
		fa = fa0;
		fb = fb0;
		max = (fa + fb + s*(b - a)) / 2;
	}
};

/* originally used for priority queue but that turns out to be really slow
class Compare{
public:
	bool operator()(const Interval a, const Interval b) const{
		return a.max < b.max;
	}
};*/

int main(int argc, const char* argv[]){
	s = 12;
	a = 1;
	b = 100;
	e = 10e-6;
	int maxLocalWork = 100;
	int localWorkToKeep = 25;
	int t = 4;
	int numNotWorking = 0;
	if (argc > 1){
		istringstream ss(argv[1]);
		ss >> t;
	}
	double max = -DBL_MAX;
	double start = omp_get_wtime();
	queue<Interval> globalWork;
	omp_lock_t workLock;
	omp_init_lock(&workLock);
	omp_lock_t maxLock;
	omp_init_lock(&maxLock);
	#pragma omp parallel num_threads(t) shared(max, maxLock, globalWork, workLock, numNotWorking)
	{
		t = omp_get_num_threads();
		int thread = omp_get_thread_num();
		queue<Interval> localWork;
		double left = a + thread*(b-a)/t;
		double right = a + (thread + 1)*(b-a)/t;
		double fa = f(left);
		double fb = f(right);
		localWork.push(Interval(left, right, fa, fb));
		double thisMax = fa > fb ? fa : fb;
		if (thisMax > max){
			omp_set_lock(&maxLock);
			if (thisMax > max){
				max = thisMax;
			}
			omp_unset_lock(&maxLock);
		}
		while (numNotWorking < t){
			while (!localWork.empty()){
				Interval current = localWork.front();
				localWork.pop();
				if (current.max > max){
					double c = (current.a + current.b) / 2;
					double fc = f(c);
					if (fc > max){
						omp_set_lock(&maxLock);
						if (fc > max){
							max = fc;
						}
						omp_unset_lock(&maxLock);
					}
					Interval left(current.a, c, current.fa, fc);
					Interval right(c, current.b, fc, current.fb);
					if (left.max > max && s*(c - current.a) >= e){
						localWork.push(left);
					}
					if (right.max > max && s*(current.b - c) >= e){
						localWork.push(right);
					}
					//too much data send it to global work
					if (localWork.size() > maxLocalWork){
						omp_set_lock(&workLock);
						while (localWork.size() > localWorkToKeep){
							globalWork.push(localWork.front());
							localWork.pop();
						}
						omp_unset_lock(&workLock);
					}
				}
			}

			//manage data since we are out
			omp_set_lock(&workLock);
			if (globalWork.size() > 0){
				while (localWork.size() < localWorkToKeep && globalWork.size() > 0){
					localWork.push(globalWork.front());
					globalWork.pop();
				}
			}
			else{
				numNotWorking++;
			}
			omp_unset_lock(&workLock);
			while (localWork.size() == 0 && numNotWorking < t){
				#pragma omp flush(globalWork, numNotWorking)
				if (globalWork.size() > 0){
					omp_set_lock(&workLock);
					if (globalWork.size() > 0){
						numNotWorking--;
						while (localWork.size() < localWorkToKeep && globalWork.size() > 0){
							localWork.push(globalWork.front());
							globalWork.pop();
						}
					}
					omp_unset_lock(&workLock);
				}
			}
		}
	}
	cout.precision(10);
	cout << "threads used " << t << endl << "time taken " << omp_get_wtime() - start << endl << "max value found " << max << endl;
}