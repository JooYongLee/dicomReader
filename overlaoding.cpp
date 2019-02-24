// tesssss.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include <iostream>
#include <vector>
#include <ctime>
#include <string>
class Timer 
{
public:
	Timer()
	:_start(0),
		_end(0),
		_elspased(0)
	{}
	~Timer() {}
	void start()
	{
		_start = clock();
	}
	void end()
	{
		_end = clock();
	}
	clock_t elapsed()
	{
		return _end - _start;
	}
	friend std::ostream& operator<<(std::ostream &out, Timer &t)
	{

		out << t.elapsed() <<","<<t.elapsed() / CLOCKS_PER_SEC << std::endl;
		return out;
	}

private:
	clock_t _start;
	clock_t _end;
	double _elspased;
		
	
};
std::vector<double> create_std(int n)
{
	std::vector<double> s;
	s.reserve(n);	
	for (int k = 0; k < n; k++)
		s.push_back(k);

	return s;
}

std::vector<double> create_std1(int n)
{
	std::vector<double> s;
	s.resize(n);
	double *ps = s.data();
	for (int k = 0; k < n; k++)
		ps[k] = k;

	return s;
}


void edit_vec(std::vector<double> &dVec)
{
	for (int k = 0; k < dVec.size(); k++)
	{
		dVec[k] = k;
	}
}
void edit_vec_copy(std::vector<double> dVec)
{
	for (int k = 0; k < dVec.size(); k++)
	{
		dVec[k] = k;
	}
}
void loop_edit_vec(std::vector<double> &dVec, int n)
{
	int iter = 0;
	while(iter++<n)
	{
		edit_vec(dVec);
		//edit_vec_copy(dVec);
	}
}

class Person
{
public:
	Person() 
		:m_strName("default")
	{
		//m_strName = "default";
	}
	//Person(const Person &p)
	//{
	//	this->m_strName = p.m_strName;
	//}
	~Person() {}
	std::string m_strName;
	friend std::ostream& operator<<(std::ostream &out, Person &t)	
	{
		out << t.m_strName << std::endl;
		return out;
	}
};

template <typename _Scalar>
class MatData
{
public:
	MatData() {}
	MatData(int row, int col)
		:pData(nullptr),
		_nWidth(col),
		_nHeight(row)
	{
		_cnt = 0;
		pData = new _Scalar[row*col];
	}
	~MatData() 
	{
		Release();
	}
	void Release()
	{
		if (pData != nullptr)
		{
			delete pData;
			pData = nullptr;
		}
	}
	friend std::ostream& operator<<(std::ostream &out, MatData &t)
	{
		_Scalar *pData = (_Scalar*)t.data();
		for (int y = 0; y < t._nHeight; y++)
		{
			for (int x = 0; x < t._nWidth; x++)
			{
				out << pData[y*t._nWidth + x] <<" ";
			}
			out << "\n";
		}
		//out << t.m_strName << std::endl;
		return out;
	}

	friend MatData& operator<<(MatData &d, _Scalar a)
	{
		d._cnt = 0;
		d.pData[d._cnt++] = a;
		return d;
	}
	friend MatData& operator,(MatData &d, _Scalar a)
	{	
		//std::cout << d._cnt << std::endl;
		d.pData[d._cnt++] = a;
		
		return d;
	}


	char* data()
	{
		return (char*)pData;
	}
public:
	int _cnt;
	int _nWidth;
	int _nHeight;
	_Scalar *pData;
};
void set_name(Person &person, const std::string &name)
{
	person.m_strName = name;
}
Person& get_peron(const std::string &name)
{
	Person p;
	p.m_strName = name;
	return p;
}
Person& get_peron(Person &p, const std::string &name)
{
	
	p.m_strName = name;
	return p;
}
void set_name2(Person person, const std::string &name)
{
	person.m_strName = name;
}
void example_stdvector_test()
{
	Timer t;
	t.start();
	std::vector<double> dVec = create_std1(3000 * 3000);
	t.end();

	t.start();
	loop_edit_vec(dVec, 10);
	t.end();
	//edit_vec(dVec);

	std::cout << t;
}
void test()
{
	Person man1;
	set_name(man1, "yosi");
	std::cout << man1;
	set_name(man1, "yo change");
	std::cout << man1;

}
//#include <stdio.h>
#include <stdarg.h>
void  printx(const char *format, ...)
{
	va_list   arg;
	int       count;

	//printf("%sERR: ", __bar);

	va_start(arg, format);
	count = vprintf(format, arg);
	va_end(arg);

	//printf("n%s", __bar);
	//exit(-1);
}
class qDebug
{
public:
	qDebug() {}
	~qDebug() {}
	friend qDebug& operator<<(qDebug &out, int a)
	{
		std::cout << a << std::endl;
		return out;
	}
	friend qDebug& operator<<(qDebug &out, std::string a)
	{				
		std::cout << a << std::endl;
		return out;
	}
	friend qDebug& operator<<(qDebug &out, const Person &p)
	{
		std::cout << p.m_strName << std::endl;
		return out;
	}
};
void mat_example()
{
	printx("에러가 발생해서 종료합니다. 에러코드= %d\n", 1234);
	MatData<int> d(2, 3);
	d << 10, -10, 3, \
		3, 2, 0;
	std::cout << d << d;
	//d.Release();

	d << 1, -1, -3, \
		23, 12, 0;

}
int main()
{
	qDebug() << 1 <<2 <<"sdfdsf";
	
	Person p; 
	

	get_peron(p, "show me the money");

	qDebug() << p;
	//std::cout << p;
	system("pause");
	
    return 0;
}

