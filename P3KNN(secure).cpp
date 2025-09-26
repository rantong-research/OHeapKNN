#include <iostream>
#include <random>
#include <chrono>
#include <queue>

#pragma GCC optimize("O3")
#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "Tools/PointerVector.h"
#include "../Tools/int.h"
#include "../Math/bigint.h"
#include "../Math/Z2k.hpp"
#include "Tools/TimerWithComm.h"
#include "Math/FixedVec.h"


using namespace std;
void test_Z2();
const int K=64;//环大小
const int k_const=5;//knn里面的k值 

string dataset_name;
string dir;
// string dataset_name="chronic";//数据集名称，自动用于后续的文件名生成
// string dataset_name="mnist";//数据集名称，自动用于后续的文件名生成

int playerno;
ez::ezOptionParser opt;
RealTwoPartyPlayer* player;
void parse_argv(int argc, const char** argv);
void gen_fake_dcf(int beta, int n);
bigint evaluate(Z2<K> x, int n,int playerID);
long long call_evaluate_time=0;


// 全局变量用于累计总运行时间
std::chrono::duration<double> total_duration(0);

class Sample{
public:
    vector<int> features;
    int label;
    Sample(int n):features(n){};
};

class KNN_party_base
{
public:
    typedef Z2<K> additive_share;
    TimerWithComm timer;
    const int nplayers=2;
    int m_playerno = 0;//player编号
    int num_features;// 特征数
    int num_train_data; // 训练集数据总量
    int num_test_data; // 测试集数据总量
    int num_label; // 训练集中label数量

    RealTwoPartyPlayer* m_player; // 通信模块
    vector<Sample*>m_sample; //训练集
    vector<Sample*>m_test; //测试集
    vector<int>m_label_list; //训练集中label的值列表

    vector< array<additive_share,2> >m_ESD_vec;
    vector<array<additive_share,2>>m_shared_label_list_count_array;
    virtual void run()=0;


    KNN_party_base(int playerNo):m_playerno(playerNo){};//构造函数

    void start_networking(ez::ezOptionParser& opt);//建立连接
    void read_meta_and_P0_sample_P1_query();
    virtual void compute_ESD_for_one_query(int idx_of_test)=0;
    virtual void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last)=0;  // top-1算法

    Z2<K> secure_compare(Z2<K>x1,Z2<K>x2,bool greater_than=true);//默认为x1>x2-->1 x1>x2-->0  x1=x2-->0 ******
    /*
        shares_selected_k为k个选择出来的share态的neighbors，label_list_count_array为二维数组，k_const*2,每一行第一个元素为出现次数，第二个元素为shares_selected_k[行表]
    */
    void label_compute(vector<Z2<K>>&shares_selected_k, vector<array<Z2<K>,2>>&label_list_count_array); 

    void compare_in_vec(vector<Z2<K>>&shares,const vector<int>compare_idx,vector<Z2<K>>&compare_res,bool greater_than); //
    void compare_in_vec(vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx,vector<Z2<K>>&compare_res,bool greater_than);

    Z2<K> reveal_one_num_to(Z2<K> x,int playerID);
    SignedZ2<K> reveal_one_num_to(SignedZ2<K> x,int playerID);

    void additive_share_data_vec(vector<Z2<K>>&shares,vector<Z2<K>>data_vec={});
    void additive_share_data_vec(vector<Z2<K>>&shares);

    /* 
    加法秘密共享的数据向量乘：
    double_res为false: v1 * v2 --> res
    double_res为true: v1.size()和res.size()一致，等于2* v2.size()    v1[:half]*v2 || v1[half:]*v2 --> res 
     */
    void mul_vector_additive( vector<Z2<K>>v1 , vector<Z2<K>>v2 , vector<Z2<K>>&res , bool double_res);

    /*
    加法秘密共享的数据标量乘法：
    res=x1*x2
    */
    void mul_additive(Z2<K>x1,Z2<K>x2,Z2<K>&res);

    /*
    secure sort in vector:
    size: compare_idx_vec一定是2的倍数，每两个为一组，表示当前需要比较的元素的索引 ，compare_res：表示比较的值，为了对齐，方便乘法运算，也为2的倍数，重复一遍。
    shares: data to be sorted，如果是二维的array也是按照第一个维度数据来进行secure sort
    compare_idx_vec: 
    compare_res：
    */
    // void SS_vec( vector<Z2<K>>&shares,const vector<int>compare_idx_vec,const vector<Z2<K>>compare_res);
    void SS_vec( vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx_vec,const vector<Z2<K>>compare_res);

    /*
        secure sort in scalar
    */
    void SS_scalar(vector<Z2<K>>&shares,int first_idx,int second_idx,bool min_then_max=true);
    void SS_scalar(vector<array<Z2<K>,2>>&shares,int first_idx,int second_idx,bool min_then_max=true);

    
    void LabelCompute_test();
    void test_cmp();

};
class KNN_party_P3KNN:public KNN_party_base
{
public:
    vector<vector<additive_share>>m_train_additive_share_vec;
    vector<vector<additive_share>>m_test_additive_share_vec;
    vector<additive_share> m_train_label_additive_share_vec;

    KNN_party_P3KNN(int playerNo):KNN_party_base(playerNo){
        std::cout<<"Entering the KNN_party_P3KNN class:"<<std::endl;
        heap.resize(k_const+1,{Z2<K>(1000000000000000000),Z2<K>(1000000000000000000)});
    }
    vector<array<Z2<K>,2>> heap;
        
    //将所有数据都转成additive share形式，包括P0的训练集数据（使用aby论文里面的share协议，进行一轮的数据发送），P1的测试集数据（使用aby论文里面的share协议，进行一轮的数据发送）
    //同时将测试集数据share，放入m_ESD_vec的第二列数据中。
    void additive_share_all_data(); 

    void compute_ESD_for_one_query(int idx_of_test);
    void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last=true);
    void run();

    void test_additive_share_all_data_function();

};

class KNN_party_optimized:public KNN_party_base
{
public:
    typedef FixedVec<Z2<K>,2> aby2_share;
    vector<vector<aby2_share>>m_train_aby2_share_vec;
    vector<vector<aby2_share>>m_test_aby2_share_vec;
    vector<vector< Z2<K> > >m_Train_Triples_0;  //P0 : num_train_data * num_features 个随机数，用于aby2 share
    vector<vector< Z2<K> > >m_Train_Triples_1;  //P1 : num_train_data * num_features 个随机数，用于aby2 share
    vector<vector< Z2<K> >>m_Test_Triples; // num_train_data * num_features  个三元组的第三个值：[(\delta_x - \delta_y)*(\delta_x - \delta_y)]
    vector< Z2<K> > m_Test_Triples_0;   // num_features 个随机数，P0用于aby2 share
    vector< Z2<K> > m_Test_Triples_1;  // num_features 个随机数，P1用于aby2 share

    KNN_party_optimized(int playerNo):KNN_party_base(playerNo){
        std::cout<<"Entering the KNN_party_optimized class:"<<std::endl;
    }

    void generate_triples_save_file(); //dealer方生成所有aby2 share随机数，自定义的三元组数据，并存入到对应文件中。属于set-up阶段，运行一次，后续就不用再运行了。
    
    void load_triples(); //读入三元组数据 分别： P0:m_Train_Triples_0 m_Train_Triples_1(aby2share的share协议) m_Test_Triples_0   P1：m_Train_Triples_1, m_Test_Triples_0, m_Test_Triples_1
    void fake_load_triples();//fake形式读入三元组数据 分别： P0:m_Train_Triples_0 m_Train_Triples_1(aby2share的share协议) m_Test_Triples_0   P1：m_Train_Triples_1, m_Test_Triples_0, m_Test_Triples_1

    void aby2_share_data_and_additive_share_label_list(); //把训练和测试数据分别在P0,P1使用aby2 share协议进行share,并且将label数据转换成加法秘密共享状态
    void aby2_share_reveal(int idx,bool is_sample_data); //测试使用，idx为reveal的样本的索引

    void additive_share_label_data();//

    Z2<K> compute_ESD_two_sample(int idx_of_sample,int idx_of_test);

    void compute_ESD_for_one_query(int idx_of_test);

    

    void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last=true);

    void run();

};


int main(int argc, const char** argv)
{
    parse_argv(argc, argv);
    // KNN_party_optimized party(playerno);

    dir="knn-1/";
    vector<string>dataset_name_list={"iris"};
    // vector<string>dataset_name_list={"Adult","Mnist","Dota2Games"};
    // vector<string>dataset_name_list={ "Toxicity", "arcene", "RNA-seq", "PEMS-SF"};
    for(int i=0;i<dataset_name_list.size();i++){
        dataset_name=dataset_name_list[i];
        std::cout<<"-----------DataSet:"<<dataset_name<<"-----------------"<<std::endl;
        KNN_party_P3KNN party(playerno);
        party.start_networking(opt);
        // std::cout<<"Network Set Up Successful ! "<<std::endl;
        party.run();
    }
    // dataset_name="Wine";
    //     cout<<"--------DataSet:"<<dataset_name<<endl;
    //     party.run();

    // dataset_name="iris";
    // cout<<"--------DataSet:"<<dataset_name<<endl;
    // party.run();
    // // party.LabelCompute_test();
    // // test_Z2();
    // dataset_name="chronic";
    // cout<<"--------DataSet:"<<dataset_name<<endl;
    // party.run();
    // party.test_cmp();
    return 0;
}

void KNN_party_P3KNN::run()
{
    read_meta_and_P0_sample_P1_query();
    std::cout<<"sample size:"<<num_train_data<<std::endl;
    std::cout<<"test size:"<<num_test_data<<std::endl;
    std::cout<<"Feature size:"<<num_features<<std::endl;
    
    // generate_triples_save_file();//这个函数必须独立运行，不能和后续load_triple一起使用。
    // cout<<"\n generate_triples_save_file success!"<<endl;

    additive_share_all_data(); //会进行一轮的通信，P0 share train data , at the same time, P1 share test data
    cout<<std::flush;
   for(register int i=0;i<num_train_data;++i)
            m_ESD_vec[i][1]=m_train_label_additive_share_vec[i];
    
    timer.start(m_player->total_comm());

    player->VirtualTwoPartyPlayer_Round=0;

    int right_prediction_cnt=0;
    for(int idx=0;idx<num_test_data;++idx)
    {
        compute_ESD_for_one_query(idx);
        struct Node { short l, r;
        bool  state, flag; };
         const int MAXQ = num_train_data; // 例如 1M，可以根据实际情况调大
     int head = 0, tail = 0;
        Node res[MAXQ];
      //  queue<Node>res;
        bool current_state=0;
    vector<Node>temp;
    
    int j=0;
    res[tail++]={0,1,false,false};
    heap[0]={m_ESD_vec[0][0],m_ESD_vec[0][1]};
    std::vector<Z2<K>> compare_res;
    std::vector<int> compare_idx;
    int gs=num_train_data-1;
         while(!(head == tail)){
        short &l=res[head].l,&r=res[head].r;
        bool &state=res[head].state,&flag=res[head].flag;
        if(current_state!=state){
          compare_res.resize(temp.size()<<1);
          compare_idx.resize(temp.size()<<1);
        for (register size_t i = 0, t = 0; i < temp.size(); ++i) {
            compare_idx[t++] = temp[i].l;
            compare_idx[t++] = temp[i].r;
        }
            compare_in_vec(heap,compare_idx,compare_res,0);
            SS_vec(heap,compare_idx,compare_res);
            temp.clear();
	    temp.emplace_back(res[head]);
            current_state^=1;
        }else if(!flag)temp.emplace_back(res[head]);
         ++head;
         if(head>=MAXQ)head -=MAXQ;
        if(r==3&&!flag&&j<gs){
            ++j;
            heap[0]={m_ESD_vec[j][0],m_ESD_vec[j][1]};
            res[tail++]={0,1,!state,false};
            if(tail>=MAXQ)tail -=MAXQ;   
            }
        if(!flag&&(r<<1)<=k_const){
        res[tail++]={r,r<<1,!state,false};
        if(tail>=MAXQ)tail -=MAXQ;   
        res[tail++]={l,r,state^1,true};
        if(tail>=MAXQ)tail -=MAXQ;  
        }
        if(flag&&(r<<1)+1<=k_const){
        res[tail++]={r,(r<<1)+1,!state,false};
        if(tail>=MAXQ)tail -=MAXQ;  
    }
    }
       if(temp.size()){
         compare_res.resize(temp.size()<<1);
         compare_idx.resize(temp.size()<<1);
        for (register size_t i = 0, t = 0; i < temp.size(); ++i) {
            compare_idx[t++] = temp[i].l;
            compare_idx[t++] = temp[i].r;
        }
            compare_in_vec(heap,compare_idx,compare_res,0);
            SS_vec(heap,compare_idx,compare_res);
       }
        vector<Z2<K>>shares_selected_k;
        for(int i=1;i<=k_const;++i)shares_selected_k.emplace_back(heap[i][1]);//,std::cout<<reveal_one_num_to(heap[i][1],0)<<" "
      //  std::cout<<endl;
        this->label_compute(shares_selected_k,m_shared_label_list_count_array);
        top_1(m_shared_label_list_count_array,k_const,false);
        Z2<K>predicted_label=reveal_one_num_to(m_shared_label_list_count_array[k_const-1][1],1); 
   //std::cout<<reveal_one_num_to(m_shared_label_list_count_array[0][0],0)<<endl;
   
   // label
   
        if(m_playerno==1)
        {
          if(Z2<K>(m_test[idx]->label)==predicted_label)
               ++right_prediction_cnt;
        }
    }

    if(m_playerno==1)std::cout<<"\n预测准确率 : "<<double(right_prediction_cnt)/(double)num_test_data<<endl;

    timer.stop(m_player->total_comm());
    cout<<"Total Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
    std::cout << "Party total time = " << timer.elapsed() << " seconds" << std::endl;
    std::cout << "Party Data sent = " << timer.mb_sent() << " MB"<<std::endl;

    std::cout<<"call_evaluate_nums : "<<call_evaluate_time<<std::endl;

    std::cout << "在Evaluation函数中 Total elapsed time: " << total_duration.count() << " seconds" << std::endl;
    call_evaluate_time=0;
    total_duration = std::chrono::duration<double>(0);
    std::cout<<"-----------------------------------------------------"<<std::endl;

}
