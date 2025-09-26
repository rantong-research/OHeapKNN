#include <iostream>
#include <random>
#include <chrono>
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
#include <stdlib.h> 

using namespace std;
const int K=64;//环大小
const int k_const=20;//knn里面的k值 

string dataset_name;
string dir;

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
        heap.resize(k_const+1,{ Z2<K>(0), Z2<K>(0) });
    }
    vector<array<Z2<K>,2>> heap;
        int heap_size=0;
         void swap(array<Z2<K>,2>& a,array<Z2<K>,2>& b){
         a[0]=a[0]^b[0];
         b[0]=b[0]^a[0];
         a[0]=a[0]^b[0];
         a[1]=a[1]^b[1];
         b[1]=b[1]^a[1];
         a[1]=a[1]^b[1];
        }
        void down(int i){
     octetStream os;
     Z2<K> tmp, x1;
    int t = i;
    
    if ((i<<1) <= heap_size) {
    x1 = secure_compare(heap[i<<1][0], heap[i][0], true);
    x1.pack(os);
    m_player->send(os);
    m_player->receive(os);
    tmp.unpack(os);
    os.clear();
    tmp = tmp + x1;
     if(tmp==1)   t <<= 1;
    }

    if ((i<<1)+1 <= heap_size) {
        x1 = secure_compare(heap[(i<<1)+1][0], heap[t][0], true);
        x1.pack(os);
        m_player->send(os);
        m_player->receive(os);
        tmp.unpack(os);
        os.clear();
        tmp = tmp + x1;
       if(tmp==1)   t = (i<<1)+1;
    }
    if (t != i) {
        swap(heap[i], heap[t]);
        down(t);
    }
}
    //将所有数据都转成additive share形式，包括P0的训练集数据（使用aby论文里面的share协议，进行一轮的数据发送），P1的测试集数据（使用aby论文里面的share协议，进行一轮的数据发送）
    //同时将测试集数据share，放入m_ESD_vec的第二列数据中。
    void additive_share_all_data(); 

    void compute_ESD_for_one_query(int idx_of_test);
    void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last=true);
    void run();

    void test_additive_share_all_data_function();

};
void KNN_party_P3KNN::run()
{
    read_meta_and_P0_sample_P1_query();
    std::cout<<"sample size:"<<num_train_data<<std::endl;
    std::cout<<"test size:"<<num_test_data<<std::endl;
    std::cout<<"Feature size:"<<num_features<<std::endl;
    // Security_heap heap(k_const);
    
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
	for(int i=1;i<=k_const;i++)heap[i]={m_ESD_vec[i-1][0],m_ESD_vec[i-1][1]};
      	for(int i=(k_const>>1);i;--i)down(i);
    
	int j=k_const;
	heap[0]={m_ESD_vec[j][0],m_ESD_vec[j][1]};
	vector<int>temp;
	Z2<K>t;
	octetStream os;
	temp.emplace_back(0);
	int gs=num_train_data-1;
    while(temp.size()){
       vector<Z2<K>>compare_res;
       vector<int>flag(k_const+1,-1),compare_idx;
        for(register int i=0;i<temp.size();++i){
            if(temp[i])compare_idx.emplace_back(temp[i]),compare_idx.emplace_back(temp[i]<<1);
        }
        if(compare_idx.size()){
            compare_res.resize(compare_idx.size());
            compare_in_vec(heap,compare_idx,compare_res,0);
            for(register int i=0;i<compare_res.size();i+=2){
               compare_res[i].pack(os);
            }
            m_player->send(os);
            m_player->receive(os);
            
            for(int i=0;i<compare_idx.size();i+=2){
            t.unpack(os);
                if(t+compare_res[i>>1]==1){
                    swap(heap[compare_idx[i]],heap[compare_idx[i+1]]);
                    flag[compare_idx[i]]=1;
                }
            }
            compare_idx.clear();
            os.clear();
        }
        for(register int i=0;i<temp.size();++i){
            if((temp[i]<<1)<k_const)compare_idx.emplace_back(temp[i]),compare_idx.emplace_back(temp[i]<<1|1);
        }
        if(compare_idx.size()){
            compare_res.resize(compare_idx.size());
            compare_in_vec(heap,compare_idx,compare_res,0);
            for(register int i=0;i<compare_res.size();i+=2){
               compare_res[i].pack(os);
            }
            m_player->send(os);
            m_player->receive(os);
            for(register int i=0;i<compare_idx.size();i+=2){
            t.unpack(os);
                if(t+compare_res[i>>1]==1){
                    swap(heap[compare_idx[i]],heap[compare_idx[i+1]]);
                    if(flag[compare_idx[i]]==1)swap(heap[compare_idx[i]<<1],heap[compare_idx[i+1]]);
                    flag[compare_idx[i]]=0;
                }
            }
            compare_idx.clear();
            os.clear();
        }
        vector<int>temp2;
        if(((temp[0]==0&&flag[0]==-1)||temp[0]!=0)&&j<gs){
        ++j;
            heap[0]={m_ESD_vec[j][0],m_ESD_vec[j][1]};
            temp2.emplace_back(0);
        }
        for(register int i=0;i<temp.size();i++){
            if(flag[temp[i]]==1&&(temp[i]<<2)<=k_const)temp2.emplace_back(temp[i]<<1);
            if(flag[temp[i]]==0&&(temp[i]<<2)+2<=k_const)temp2.emplace_back(temp[i]<<1|1);
    }
    temp=temp2;
     
}

           vector<Z2<K>>shares_selected_k;
        for(int i=1;i<=k_const;++i){
        shares_selected_k.emplace_back(heap[i][1]);
        }
        
           this->label_compute(shares_selected_k,m_shared_label_list_count_array);
        top_1(m_shared_label_list_count_array,k_const,false);
        Z2<K>predicted_label=reveal_one_num_to(m_shared_label_list_count_array[k_const-1][1],1); 


        if(m_playerno)
        {
      
            if(Z2<K>(m_test[idx]->label)==predicted_label)
                right_prediction_cnt++;
        }


        // for(int i=0;i<k_const;++i)
        // {
        //     Z2<K>tmp=heap.heap[1].index;
        //     if(m_playerno)
        //     {
        //         if(m_test[idx]->label==reveal_one_num_to(m_train_label_additive_share_vec[tmp],playerno))
        //             right_prediction_cnt++;
        //     }
        // }

    }

    if(m_playerno)std::cout<<"\n预测准确率 : "<<double(right_prediction_cnt)/(double)num_test_data<<endl;

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