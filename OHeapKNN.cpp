/*
 * This implementation is developed based on the open-source Garnet framework.
 * https://github.com/FudanMPL/Garnet
 *
 * We extend the original system to implement a complete simulation of the
 * OHeapKNN scheme, which leverages an oblivious heap to support
 * privacy-preserving KNN computation.
 *
 * The code includes secure computation protocols, data sharing procedures,
 * and the full execution pipeline for evaluating the OHeapKNN scheme.
 * In particular, the oblivious heap design ensures that access patterns
 * during heap operations are fully hidden.
 *
 * Our implementation enables end-to-end secure KNN computation under
 * the OHeapKNN design within the Garnet environment, providing both
 * strong privacy guarantees and improved efficiency.
 */

 #include <iostream>
 #include <random>
 #include <chrono>
 #include <queue>
 
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
 const int K=64;// Ring size
 const int k_const=5;// k value in KNN 
 
 string dataset_name;
 string dir;
 
 int playerno;
 ez::ezOptionParser opt;
 RealTwoPartyPlayer* player;
 void parse_argv(int argc, const char** argv);
 void gen_fake_dcf(int beta, int n);
 bigint evaluate(Z2<K> x, int n,int playerID);
 long long call_evaluate_time=0;
 
 
 // Global variable used to accumulate total runtime
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
     int m_playerno = 0;// player ID
     int num_features;// Number of features
     int num_train_data; // Total number of training samples
     int num_test_data; // Total number of test samples
     int num_label; // Number of labels in the training set
 
     RealTwoPartyPlayer* m_player; // Communication module
     vector<Sample*>m_sample; // Training set
     vector<Sample*>m_test; // Test set
     vector<int>m_label_list; // Training set label list
 
     vector< array<additive_share,2> >m_ESD_vec;
     vector<array<additive_share,2>>m_shared_label_list_count_array;
     virtual void run()=0;
 
 
     KNN_party_base(int playerNo):m_playerno(playerNo){};// Constructor
 
     void start_networking(ez::ezOptionParser& opt);// Establish connection
     void read_meta_and_P0_sample_P1_query();
     virtual void compute_ESD_for_one_query(int idx_of_test)=0;
     virtual void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last)=0;  // Top-1 algorithm
 
     Z2<K> secure_compare(Z2<K>x1,Z2<K>x2,bool greater_than=true);// Default: x1 > x2 -> 1, x1 < x2 -> 0, x1 == x2 -> 0 ******
     /*
         shares_selected_k contains the k selected neighbors in shared form. label_list_count_array is a 2D array of size k_const * 2. In each row, the first element is the occurrence count, and the second element is shares_selected_k[row].
     */
     void label_compute(vector<Z2<K>>&shares_selected_k, vector<array<Z2<K>,2>>&label_list_count_array); 
 
     void compare_in_vec(vector<Z2<K>>&shares,const vector<int>compare_idx,vector<Z2<K>>&compare_res,bool greater_than); //
     void compare_in_vec(vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx,vector<Z2<K>>&compare_res,bool greater_than);
 
     Z2<K> reveal_one_num_to(Z2<K> x,int playerID);
     SignedZ2<K> reveal_one_num_to(SignedZ2<K> x,int playerID);
 
     void additive_share_data_vec(vector<Z2<K>>&shares,vector<Z2<K>>data_vec={});
     void additive_share_data_vec(vector<Z2<K>>&shares);
 
     /* 
     Vector multiplication under additive secret sharing:
     double_res = false: v1 * v2 --> res
     double_res = true: v1.size() equals res.size(), both equal to 2 * v2.size(); v1[:half] * v2 || v1[half:] * v2 --> res 
      */
     void mul_vector_additive( vector<Z2<K>>v1 , vector<Z2<K>>v2 , vector<Z2<K>>&res , bool double_res);
 
     /*
     Scalar multiplication under additive secret sharing:
     res=x1*x2
     */
     void mul_additive(Z2<K>x1,Z2<K>x2,Z2<K>&res);
 
     /*
     secure sort in vector:
     size: compare_idx_vec must be even. Every two indices form a pair indicating the indices of the elements to compare. compare_res stores the comparison results. For alignment and easier multiplication, it is also duplicated to an even length.
     shares: data to be sorted. If it is a 2D array, secure sort is also performed according to the data in the first dimension.
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
 class KNN_party_OHeapKNN:public KNN_party_base
 {
 public:
     vector<vector<additive_share>>m_train_additive_share_vec;
     vector<vector<additive_share>>m_test_additive_share_vec;
     vector<additive_share> m_train_label_additive_share_vec;
 
     KNN_party_OHeapKNN(int playerNo):KNN_party_base(playerNo){
         std::cout<<"Entering the KNN_party_OHeapKNN class:"<<std::endl;
         heap.resize(k_const+1,{Z2<K>(1000000000000000000),Z2<K>(1000000000000000000)});
     }
     vector<array<Z2<K>,2>> heap;
         
     // Convert all data into additive shares, including P0's training data (using the sharing protocol in the ABY paper, with one communication round) and P1's test data (also using the sharing protocol in the ABY paper, with one communication round)
     // At the same time, share the test data and place it into the second column of m_ESD_vec.
     void additive_share_all_data(); 
 
     void compute_ESD_for_one_query(int idx_of_test);
     void top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last=true);
     void run();
 
     void test_additive_share_all_data_function();
 
 };
 
 
 
 int main(int argc, const char** argv)
 {
     parse_argv(argc, argv);
 
 
     dir="knn-1/";
     vector<string>dataset_name_list={"iris"};
     // vector<string>dataset_name_list={"Adult","Mnist","Dota2Games"};
     // vector<string>dataset_name_list={ "Toxicity", "arcene", "RNA-seq", "PEMS-SF"};
     for(int i=0;i<dataset_name_list.size();i++){
         dataset_name=dataset_name_list[i];
         std::cout<<"-----------DataSet:"<<dataset_name<<"-----------------"<<std::endl;
         KNN_party_OHeapKNN party(playerno);
         party.start_networking(opt);
         // std::cout<<"Network Set Up Successful ! "<<std::endl;
         party.run();
     }
     return 0;
 }
 
 
 
 
 void KNN_party_base::start_networking(ez::ezOptionParser& opt) 
 {
     string hostname, ipFileName;
     int pnbase;
     int my_port;
     opt.get("--portnumbase")->getInt(pnbase);
     opt.get("--hostname")->getString(hostname);
     opt.get("--ip-file-name")->getString(ipFileName);
     ez::OptionGroup* mp_opt = opt.get("--my-port");
     if (mp_opt->isSet)
       mp_opt->getInt(my_port);
     else
       my_port = Names::DEFAULT_PORT;
 
     Names playerNames;
 
     if (ipFileName.size() > 0) {
       if (my_port != Names::DEFAULT_PORT)
         throw runtime_error("cannot set port number when using IP file");
       playerNames.init(playerno, pnbase, ipFileName, nplayers);
     } else {
       Server::start_networking(playerNames, playerno, nplayers,
                               hostname, pnbase, my_port);
     }
     this->m_player = new RealTwoPartyPlayer(playerNames, 1-playerno, 0);
     player = this->m_player;
   }
 
 
 
 Z2<K> KNN_party_base::reveal_one_num_to(Z2<K> x,int playerID)
 {
     octetStream os;
     if(playerno==playerID)
     {
         m_player->receive(os);
         Z2<K>tmp;
         tmp.unpack(os);
         return tmp+x;
     }
     else
     {
         x.pack(os);
         m_player->send(os);
         return x;
     }
 }
 
 SignedZ2<K> KNN_party_base::reveal_one_num_to(SignedZ2<K> x,int playerID)
 {
     octetStream os;
     if(playerno==playerID)
     {
         m_player->receive(os);
         SignedZ2<K>tmp;
         tmp.unpack(os);
         return tmp+x;
     }
     else
     {
         x.pack(os);
         m_player->send(os);
         return x;
     }
 }
 struct DistanceIndex{
     Z2<K>dist;
     Z2<K>index;
     DistanceIndex(Z2<K>dist,int index):dist(dist),index(index){}
 };
 
 void KNN_party_OHeapKNN::run()
 {
     read_meta_and_P0_sample_P1_query();
     std::cout<<"sample size:"<<num_train_data<<std::endl;
     std::cout<<"test size:"<<num_test_data<<std::endl;
     std::cout<<"Feature size:"<<num_features<<std::endl;
     
     // generate_triples_save_file();// This function must be run independently and cannot be used together with the subsequent load_triple.
     // cout<<"\n generate_triples_save_file success!"<<endl;
 
     additive_share_all_data(); // This performs one communication round: P0 shares the training data, while P1 shares the test data at the same time.
     cout<<std::flush;
    for(int i=0;i<num_train_data;++i)
             m_ESD_vec[i][1]=m_train_label_additive_share_vec[i];
     
     timer.start(m_player->total_comm());
 
     player->VirtualTwoPartyPlayer_Round=0;
 
     int right_prediction_cnt=0;
     
     
     
     for(int idx=0;idx<num_test_data;++idx)
     {
        compute_ESD_for_one_query(idx);
         struct Node { short l, r;
         bool  state, flag; };
          const int MAXQ = num_train_data; 
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
         for (size_t i = 0, t = 0; i < temp.size(); ++i) {
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
         for (size_t i = 0, t = 0; i < temp.size(); ++i) {
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
         // 
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
 
     if(m_playerno==1)std::cout<<"\nPrediction accuracy : "<<double(right_prediction_cnt)/(double)num_test_data<<endl;
 
     timer.stop(m_player->total_comm());
     cout<<"Total Round count = "<<player->VirtualTwoPartyPlayer_Round<< " online round"<<endl;
     std::cout << "Party total time = " << timer.elapsed() << " seconds" << std::endl;
     std::cout << "Party Data sent = " << timer.mb_sent() << " MB"<<std::endl;
 
     std::cout<<"call_evaluate_nums : "<<call_evaluate_time<<std::endl;
 
     std::cout << "Total elapsed time in the Evaluation function: " << total_duration.count() << " seconds" << std::endl;
     call_evaluate_time=0;
     total_duration = std::chrono::duration<double>(0);
     std::cout<<"-----------------------------------------------------"<<std::endl;
 
 }
 
 
 void KNN_party_base::label_compute(vector<Z2<K>>&share_k_neighbors, vector<array<Z2<K>,2>>&label_list_count_array)
 {
     label_list_count_array.resize(k_const);
     vector<vector<Z2<K>>> cmp_2d_vec_res(k_const,vector<Z2<K>>(k_const));
     vector<int>cmp_vec_idx;
     for(int i=0;i<k_const;++i){
         for(int j=0;j<k_const;++j){
             cmp_vec_idx.emplace_back(i);
             cmp_vec_idx.emplace_back(j);
         }
     }
     vector<Z2<K>>cmp_res_vec(cmp_vec_idx.size());
     compare_in_vec(share_k_neighbors,cmp_vec_idx,cmp_res_vec,true);
     int g =cmp_res_vec.size()>>1;
     for(int i=0;i<g;++i){
         cmp_2d_vec_res[cmp_vec_idx[i<<1]][cmp_vec_idx[(i<<1)+1]]=Z2<K>(m_playerno)-cmp_res_vec[i<<1];
     }
 
     vector<Z2<K>>v1(k_const*k_const),v2(k_const*k_const),res(k_const*k_const);
     for(int i=0;i<k_const;++i){
         for(int j=0;j<k_const;++j){
             v1[i*k_const+j]=cmp_2d_vec_res[i][j];
             v2[i*k_const+j]=cmp_2d_vec_res[j][i];
         }
     }
     mul_vector_additive(v1,v2,res,false);
 
     // vector<array<Z2<K>,2>>label_list_count_array(k_const);
     for(int i=0;i<k_const;++i){
         Z2<K>tmp(0);
         for(int j=0;j<k_const;++j){
             tmp+=res[i*k_const+j];
         }
         label_list_count_array[i]={tmp,share_k_neighbors[i]};
     }
     // cout<<"Test result of label_list_count_array (frequency  label):"<<endl;
     // for(int i=0;i<k_const;i++){
     //     Z2<K>tmp_0=reveal_one_num_to(label_list_count_array[i][0],1);// Occurrence frequency
     //     Z2<K>tmp_1=reveal_one_num_to(label_list_count_array[i][1],1);// Corresponding label
     //     if(playerno==1){
     //         std::cout<<tmp_0<<"  "<<tmp_1<<endl;
     //     }
     // }
 
 
 
     // for(int i=0;i<num_label;i++)
     // {
     //     Z2<K>tmp(0);
     //     for(int j=0;j<k_const;j++)
     //     {
     //         // Z2<K>u1=Z2<K>(m_playerno)-this->secure_compare(label_list_count_array[i][1],shares_selected_k[j]);
     //         // Z2<K>u2=Z2<K>(m_playerno)-this->secure_compare(shares_selected_k[j],label_list_count_array[i][1]);
     //         vector<Z2<K>>U(4),value_tmp={label_list_count_array[i][1],shares_selected_k[j],shares_selected_k[j],label_list_count_array[i][1]};
     //         compare_in_vec(value_tmp,{0,1,2,3},U,true);
     //         Z2<K>tmp_res;
     //         mul_additive(Z2<K>(m_playerno)-U[0],Z2<K>(m_playerno)-U[2],tmp_res);// This step, Z2<K>(m_playerno)-, is critical; otherwise all results will be incorrect.
     //         tmp+=tmp_res;
     //     }
     //     label_list_count_array[i][0]=tmp;
     // }
     
 }
 
 void KNN_party_base::compare_in_vec(vector<Z2<K>>&shares,const vector<int>compare_idx_vec,vector<Z2<K>>&compare_res,bool greater_than)
 {
     assert(compare_idx_vec.size()&&compare_idx_vec.size()==compare_res.size());
     bigint r_tmp;
     fstream r;
     r.open("Player-Data/2-fss/r" + to_string( m_playerno), ios::in);
     r >> r_tmp;
     r.close();
     SignedZ2<K>alpha_share=(SignedZ2<K>)r_tmp;
     // cout<<"\n bigint:"<<r_tmp<<"  Z2<K>: " << alpha_share<<endl;
     int size_res=compare_idx_vec.size()/2;
 
     vector<SignedZ2<K>>compare_res_t(compare_res.size());
     if(greater_than)
     {
         for(int i=0;i<size_res;i++)
         {
             compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[(i<<1)+1]])-SignedZ2<K>(shares[compare_idx_vec[i<<1]])+alpha_share;
             //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
         }
     }
     else{
         for(int i=0;i<size_res;i++)
         {
             compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[i<<1]])-SignedZ2<K>(shares[compare_idx_vec[(i<<1)+1]])+alpha_share;
             //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
         }
     }
 
        
     vector<SignedZ2<K>>tmp_res(size_res);
 
     octetStream send_os,receive_os;
     for(int i=0;i<size_res;i++)compare_res_t[i].pack(send_os);
     player->send(send_os);
     player->receive(receive_os);
     for(int i=0;i<size_res;i++)
     {
         SignedZ2<K>ttmp;
         ttmp.unpack(receive_os);
         tmp_res[i]=compare_res_t[i]+ttmp;
     }
 
     for(int i=0;i<size_res;i++)
     {
         bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
         SignedZ2<K> dcf_u,dcf_v;
         dcf_res_u = evaluate(tmp_res[i], K,m_playerno);
         tmp_res[i] += 1LL<<(K-1);
         dcf_res_v = evaluate(tmp_res[i], K,m_playerno);
         auto size = dcf_res_u.get_mpz_t()->_mp_size;
         mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
         if(size < 0)
             dcf_u = -dcf_u;
         size = dcf_res_v.get_mpz_t()->_mp_size;
         mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
         if(size < 0)
             dcf_v = -dcf_v;
         if(tmp_res[i].get_bit(K-1)){
             r_tmp = dcf_v - dcf_u + m_playerno;
         }
         else{
             r_tmp = dcf_v - dcf_u;
         }
         compare_res[i<<1]=SignedZ2<K>(m_playerno)-r_tmp;
 
         // compare_res[2*i]=evaluate(tmp_res[i],K,m_playerno);
         compare_res[(i<<1)+1]=compare_res[i<<1];// Repeat once   
     }
 
 }
 
 
 
 
 void KNN_party_base::compare_in_vec(vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx_vec,vector<Z2<K>>&compare_res,bool greater_than)
 {
     assert(compare_idx_vec.size()&&compare_idx_vec.size()==compare_res.size());
     bigint r_tmp;
     fstream r;
     r.open("Player-Data/2-fss/r" + to_string( m_playerno), ios::in);
     r >> r_tmp;
     r.close();
     SignedZ2<K>alpha_share=(SignedZ2<K>)r_tmp;
     // cout<<"\n bigint:"<<r_tmp<<"  Z2<K>: " << alpha_share<<endl;
     int size_res=compare_idx_vec.size()>>1;
 
 
     vector<SignedZ2<K>>compare_res_t(compare_res.size());
     if(greater_than)
     {
         for(int i=0;i<size_res;++i)
         {
             compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[(i<<1)+1]][0])-SignedZ2<K>(shares[compare_idx_vec[i<<1]][0])+alpha_share;
             //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
         }
     }
     else{
         for(int i=0;i<size_res;++i)
         {
             compare_res_t[i]=SignedZ2<K>(shares[compare_idx_vec[i<<1]][0])-SignedZ2<K>(shares[compare_idx_vec[(i<<1)+1]][0])+alpha_share;
             //  cout<<reveal_one_num_to(shares[compare_idx_vec[2*i]],0)<<" "<<reveal_one_num_to(shares[compare_idx_vec[2*i+1]],0)<<endl;
         }
     }
     
     // cout<<endl;
        
     vector<SignedZ2<K>>tmp_res(size_res);
 
     octetStream send_os,receive_os;
     for(int i=0;i<size_res;++i)compare_res_t[i].pack(send_os);
     player->send(send_os);
     player->receive(receive_os);
     for(int i=0;i<size_res;++i)
     {
         SignedZ2<K>ttmp;
         ttmp.unpack(receive_os);
         tmp_res[i]=compare_res_t[i]+ttmp;
     }
 
     for(int i=0;i<size_res;++i)
     {
         bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
         SignedZ2<K> dcf_u,dcf_v;
         dcf_res_u = evaluate(tmp_res[i], K,  m_playerno);
         tmp_res[i] += 1LL<<(K-1);
         dcf_res_v = evaluate(tmp_res[i], K, m_playerno);
         auto size = dcf_res_u.get_mpz_t()->_mp_size;
         mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
         if(size < 0)
             dcf_u = -dcf_u;
         size = dcf_res_v.get_mpz_t()->_mp_size;
         mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
         if(size < 0)
             dcf_v = -dcf_v;
         if(tmp_res[i].get_bit(K-1)){
             r_tmp = dcf_v - dcf_u + m_playerno;
         }
         else{
             r_tmp = dcf_v - dcf_u;
         }
         compare_res[i<<1]=SignedZ2<K>(m_playerno)-r_tmp;
 
         // compare_res[2*i]=evaluate(tmp_res[i],K,m_playerno);
         compare_res[(i<<1)+1]=compare_res[i<<1];// Repeat once   
     }
 
 }
 
 
 void KNN_party_OHeapKNN::top_1(vector<array<Z2<K>,2>>&shares,int size_of_need_select,bool min_in_last)
 {
     for(int i=0;i<size_of_need_select-1;i++)
         SS_scalar(shares, i, size_of_need_select-1, !min_in_last);
 }
 
 
 
 
 
 void KNN_party_base::SS_scalar(vector<array<Z2<K>,2>>&shares,int first_idx,int second_idx,bool min_then_max)
 {
     Z2<K>u=secure_compare(shares[first_idx][0],shares[second_idx][0],min_then_max);
     vector<Z2<K>> Y(4);
     // mul_additive(u,shares[first_idx][0],y1);
     // mul_additive(u,shares[second_idx][0],y2);
     mul_vector_additive({shares[first_idx][0],shares[second_idx][0],shares[first_idx][1],shares[second_idx][1]},{u,u},Y,true);
     shares[first_idx][0]=shares[first_idx][0]-Y[0]+Y[1];
     shares[second_idx][0]=shares[second_idx][0]+Y[0]-Y[1];
 
     // mul_additive(u,shares[first_idx][1],y1);
     // mul_additive(u,shares[second_idx][1],y2);
     shares[first_idx][1]=shares[first_idx][1]-Y[2]+Y[3];
     shares[second_idx][1]=shares[second_idx][1]+Y[2]-Y[3];
 }
 
 
 void KNN_party_base::SS_scalar(vector<Z2<K>>&shares,int first_idx,int second_idx,bool min_then_max)
 {
     Z2<K>u=secure_compare(shares[first_idx],shares[second_idx],min_then_max);
     vector<Z2<K>> Y(2);
     mul_vector_additive({shares[first_idx],shares[second_idx]},{u,u},Y,false);
     shares[first_idx]=shares[first_idx]-Y[0]+Y[1];
     shares[second_idx]=shares[second_idx]+Y[0]-Y[1];
 
 }
 
 // void KNN_party_base::SS_vec( vector<Z2<K>>&shares,const vector<int>compare_idx_vec,const vector<Z2<K>>compare_res)
 // {
 //     assert(compare_idx_vec.size()&&compare_idx_vec.size()==compare_res.size());
 //     int size_of_cur_cmp=compare_idx_vec.size();
 //     vector<Z2<K>>tmp_ss;
 //     for(int i=0;i<size_of_cur_cmp;i++)tmp_ss.push_back(shares[compare_idx_vec[i]]);
 //     vector<Z2<K>>tmp_res(size_of_cur_cmp);
 //     mul_vector_additive(tmp_ss,compare_res,tmp_res,false);
 //     for(int i=0;i<size_of_cur_cmp/2;i++)
 //     {
 //         tmp_ss[2*i]=tmp_ss[2*i]-tmp_res[2*i]+tmp_res[2*i+1];
 //         tmp_ss[2*i+1]=tmp_ss[2*i+1] + tmp_res[2*i]-tmp_res[2*i+1];
 //     }
 //     for(int i=0;i<size_of_cur_cmp;i++)shares[compare_idx_vec[i]]=tmp_ss[i];
 // }
 
 
 void KNN_party_base::SS_vec( vector<array<Z2<K>,2>>&shares,const vector<int>compare_idx_vec,const vector<Z2<K>>compare_res)
 {
     assert(compare_idx_vec.size()&&compare_idx_vec.size()==compare_res.size());
     int size_of_cur_cmp=compare_idx_vec.size();
     vector<Z2<K>>tmp_ss;
     for(int i=0;i<size_of_cur_cmp;++i)tmp_ss.emplace_back(shares[compare_idx_vec[i]][0]);
     for(int i=0;i<size_of_cur_cmp;++i)tmp_ss.emplace_back(shares[compare_idx_vec[i]][1]);
     vector<Z2<K>>tmp_res(size_of_cur_cmp<<1);// The first half stores the multiplication results for x, and the second half stores the multiplication results for labels
     mul_vector_additive(tmp_ss,compare_res,tmp_res,true);
     int h=size_of_cur_cmp>>1;
     for(int i=0;i<h;++i)
     {
         tmp_ss[i<<1]=tmp_ss[i<<1]-tmp_res[i<<1]+tmp_res[(i<<1)+1]; // If the comparison result is x1 > x2 -> 1, the order becomes [x_min, x_max]
         tmp_ss[(i<<1)+1]=tmp_ss[(i<<1)+1] + tmp_res[i<<1]-tmp_res[(i<<1)+1];
     }
     for(int i=0;i<size_of_cur_cmp;++i)shares[compare_idx_vec[i]][0]=tmp_ss[i];
     h=size_of_cur_cmp>>1;
     for(int i=0;i<h;++i)// tmp_ss stores the values collected from shares according to compare_idx_vec, and tmp_res stores the comparison results; both have twice the comparison-result length
     {
         tmp_ss[(i<<1)+size_of_cur_cmp]=tmp_ss[(i<<1)+size_of_cur_cmp]-tmp_res[(i<<1)+size_of_cur_cmp]+tmp_res[(i<<1)+1+size_of_cur_cmp];
         tmp_ss[(i<<1)+1+size_of_cur_cmp]=tmp_ss[(i<<1)+1+size_of_cur_cmp] + tmp_res[(i<<1)+size_of_cur_cmp]-tmp_res[(i<<1)+1+size_of_cur_cmp];
     }
     for(int i=0;i<size_of_cur_cmp;++i)shares[compare_idx_vec[i]][1]=tmp_ss[i+size_of_cur_cmp];
 }
 
 
 void KNN_party_base::mul_additive(Z2<K>x1,Z2<K>x2,Z2<K>&res)
 {
     Z2<K>a(0),b(0),c(0);
     octetStream send_os,receive_os;
     (x1-a).pack(send_os);
     (x2-b).pack(send_os);
     player->send(send_os);
     player->receive(receive_os);
     Z2<K>tmp_e,tmp_f;
     tmp_e.unpack(receive_os);
     tmp_f.unpack(receive_os);
 
     Z2<K>e=tmp_e+x1-a;
     Z2<K>f=tmp_f+x2-b;
     Z2<K>r=f*a+e*b+c;;
     if(player->my_num())
         r=r+e*f;
     res=r;
 }
 
 void KNN_party_base::mul_vector_additive( vector<Z2<K>>v1 , vector<Z2<K>>v2 , vector<Z2<K>>&res , bool double_res)
 {
     if(double_res)
     {
         assert(v1.size()==v2.size()*2&&v1.size()==res.size());
         Z2<K>a(0),b(0),c(0);
         octetStream send_os,receive_os;
         int half_size=v2.size();
         for(int i=0;i<half_size;++i)
         {
             (v1[i]-a).pack(send_os);
             (v2[i]-b).pack(send_os);
         }
         for(int i=0;i<half_size;++i)
         {
             (v1[i+half_size]-a).pack(send_os);
             (v2[i]-b).pack(send_os);
         }
         player->send(send_os);
         player->receive(receive_os);
         vector<Z2<K>>tmp(v1.size()<<1);
         for(int i=0;i<half_size;++i)
         {
             tmp[i<<1].unpack(receive_os);
             tmp[(i<<1)+1].unpack(receive_os);
             tmp[i<<1]=tmp[i<<1]+v1[i]-a;
             tmp[(i<<1)+1]= tmp[(i<<1)+1]+v2[i]-b;
         }
         for(int i=0;i<half_size;++i)
         {   
             Z2<K>e=tmp[i<<1];
             Z2<K>f=tmp[(i<<1)+1];
             Z2<K>r=f*a+e*b+c;;
             if(player->my_num())
                 r=r+e*f;
             res[i]=r;
         }
 
         for(int i=0;i<half_size;++i)
         {
             tmp[i<<1].unpack(receive_os);
             tmp[(i<<1)+1].unpack(receive_os);
             tmp[i<<1]=tmp[i<<1]+v1[i+half_size]-a;
             tmp[(i<<1)+1]= tmp[(i<<1)+1]+v2[i]-b;
         }
         for(int i=0;i<half_size;++i)
         {   
             Z2<K>e=tmp[i<<1];
             Z2<K>f=tmp[(i<<1)+1];
             Z2<K>r=f*a+e*b+c;;
             if(player->my_num())
                 r=r+e*f;
             res[i+half_size]=r;
         }
 
 
     }
     else{
         assert(v1.size()==v2.size());
         Z2<K>a(0),b(0),c(0);
         octetStream send_os,receive_os;
         for(int i=0;i<(int)v1.size();++i)
         {
             (v1[i]-a).pack(send_os);
             (v2[i]-b).pack(send_os);
         }
         player->send(send_os);
         player->receive(receive_os);
         vector<Z2<K>>tmp(v1.size()<<1);
         for(int i=0;i<(int)v1.size();++i)
         {
             tmp[i<<1].unpack(receive_os);
             tmp[(i<<1)+1].unpack(receive_os);
             tmp[i<<1]=tmp[i<<1]+v1[i]-a;
             tmp[(i<<1)+1]= tmp[(i<<1)+1]+v2[i]-b;
         }
         for(int i=0;i<(int)v1.size();++i)
         {   
             Z2<K>&e=tmp[i<<1];
             Z2<K>&f=tmp[(i<<1)+1];
             Z2<K>r=f*a+e*b+c;;
             if(player->my_num())
                 r=r+e*f;
             res[i]=r;
         }
     }
     
 }
 
 Z2<K> KNN_party_base::secure_compare(Z2<K>x1,Z2<K>x2,bool greater_than)//x1>x2-->1   x1<x2-->0   x1==x2-->0
 {
     // cout<<x1<<" "<<x2<<endl;
     bigint r_tmp;
     fstream r;
     const static string path = "Player-Data/2-fss/r" + to_string(m_playerno);
     r.open(path, ios::in);
     r >> r_tmp;
     r.close();
     SignedZ2<K>alpha_share=(SignedZ2<K>)r_tmp;
     SignedZ2<K>revealed=SignedZ2<K>(x2)-SignedZ2<K>(x1)+alpha_share;
     if(greater_than==false){
         revealed=SignedZ2<K>(x1)-SignedZ2<K>(x2)+alpha_share;
     }
     
     octetStream send_os,receive_os;
     revealed.pack(send_os);
     player->send(send_os);
     player->receive(receive_os);
     SignedZ2<K>ttmp;
     ttmp.unpack(receive_os);
     revealed+=ttmp;
 
     static bigint dcf_res_u, dcf_res_v, dcf_res, dcf_res_reveal;
     static SignedZ2<K> dcf_u,dcf_v;
     dcf_res_u = evaluate(revealed, K,m_playerno);
     revealed += 1LL<<(K-1);
     dcf_res_v = evaluate(revealed, K,m_playerno);
     auto size = dcf_res_u.get_mpz_t()->_mp_size;
     mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
     if(size < 0)
         dcf_u = -dcf_u;
     size = dcf_res_v.get_mpz_t()->_mp_size;
     mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
     if(size < 0)
         dcf_v = -dcf_v;
     if(revealed.get_bit(K-1)){
         r_tmp = dcf_v - dcf_u + m_playerno;
     }
     else{
         r_tmp = dcf_v - dcf_u;
     }
     SignedZ2<K>res=SignedZ2<K>(m_playerno)-r_tmp;
     // std::cout<<"revealed secure_compare result :"<<reveal_one_num_to(Z2<K>(res),0)<<std::endl;
     return Z2<K>(res);
 
 }
 
 void KNN_party_OHeapKNN::compute_ESD_for_one_query(int idx_of_test)
 {
     // cout<<"Enter compute_ESD_for_one_query"<<endl;
      if(int(m_ESD_vec.size())!=num_train_data)
         m_ESD_vec.resize(num_train_data);
     std::vector<Z2<K>> Z(num_train_data * num_features,Z2<K>(0));
     Z2<K>r(0),r_square(0),tmp(0);// Initialized to 0 by default
 
     octetStream send_os,receive_os;
      int base =0;
     for(int i=0;i<num_train_data;++i,base+=num_features)
     {
     std::vector<Z2<K>>& train_row = m_train_additive_share_vec[i];
     std::vector<Z2<K>>& test_row  = m_test_additive_share_vec[idx_of_test];
     for(int j = 0; j < num_features; ++j){
         Z[base + j] = train_row[j] - test_row[j] + r;
         Z[base + j].pack(send_os);
     }
     }
     m_player->send(send_os);
 
     m_player->receive(receive_os);
     base=0;
     for(int i=0;i<num_train_data;++i,base+=num_features)
     {	
         for(int j(0);j<num_features;++j)
         {
             tmp.unpack(receive_os);
             Z[base + j] =Z[base + j]+  tmp ;
         }
     }
     base=0;
     for(int i=0;i<num_train_data;++i,base+=num_features)
     {
         tmp=Z2<K>(0); 
         for( int j(0);j<num_features;++j)
         {
 
             tmp=tmp- (Z[base + j]*r<<1) + r_square;
             if(m_playerno)tmp=tmp+Z[base + j]*Z[base + j];
         }
         m_ESD_vec[i][0]=tmp;
     }
     // cout<<"compute_ESD_for_one_query ended!"<<endl;
 }
 
 
 void KNN_party_OHeapKNN::test_additive_share_all_data_function()
 {
     Z2<K>tmp_0(0);
     for(int i=0;i<10;i++)
     {
         for(int j=0;j<num_features;j++)
         {
             tmp_0=reveal_one_num_to(m_train_additive_share_vec[i][j],0);
             if(m_playerno==0)assert(tmp_0==Z2<K>(m_sample[i]->features[j]));
         }
     }
     std::cout<<std::endl;
     std::cout<<std::endl;
 
     for(int i=0;i<10;i++)
     {
         for(int j=0;j<num_features;j++)
         {
             tmp_0=reveal_one_num_to(m_test_additive_share_vec[i][j],0);
             if(m_playerno==0)assert(tmp_0==Z2<K>(m_test[i]->features[j]));
         }
     }
     std::cout<<std::endl;
     std::cout<<std::endl;
     for(int i=0;i<num_train_data;i++)
     {
         tmp_0=reveal_one_num_to(m_ESD_vec[i][1],0);
         if(m_playerno==0) assert(tmp_0==Z2<K>(m_sample[i]->label));
     }
     cout<<"test_additive_share_all_data_function() ended!";
     
 
 }
 
 
 void KNN_party_OHeapKNN::additive_share_all_data()
 {
     m_train_additive_share_vec.resize(num_train_data);
     for(int i=0;i<num_train_data;++i) 
         m_train_additive_share_vec[i].resize(num_features);
 
     m_test_additive_share_vec.resize(num_test_data);
     for(int i=0;i<num_test_data;++i)
         m_test_additive_share_vec[i].resize(num_features);
 
     m_train_label_additive_share_vec.resize(num_train_data);
     m_ESD_vec.resize(num_train_data);
 
     if(playerno==0)
     {
         octetStream os;
         PRNG prng;
         prng.ReSeed();
         Z2<K>random_data;
         for(int i=0;i<num_train_data;++i)
         {
             for(int j=0;j<num_features;++j)
             {   
                 
                 random_data.randomize(prng);
                 m_train_additive_share_vec[i][j]=Z2<K>(m_sample[i]->features[j])-random_data;
 
                 random_data.pack(os);
 
             }
 
             random_data.randomize(prng);
             m_train_label_additive_share_vec[i]=Z2<K>(m_sample[i]->label)-random_data;
             random_data.pack(os);
 
         }
         m_player->send(os);
         // cout<<"Train data additive_share sending ended!"<<endl;
 
         os.clear();
         m_player->receive(os);
         for(int i=0;i<num_test_data;++i)
         {
             for(int j=0;j<num_features;++j)
             {
                 m_test_additive_share_vec[i][j].unpack(os);
             }
         }
         // cout<<"Test data additive_share receiving ended!"<<endl;
 
     }
     else
     {
         octetStream os;
         PRNG prng;
         prng.ReSeed();
         Z2<K>random_data;
 
         m_player->receive(os);
         for(int i=0;i<num_train_data;++i)
         {
             for(int j=0;j<num_features;++j)
             {
                 m_train_additive_share_vec[i][j].unpack(os);
             }
             m_train_label_additive_share_vec[i].unpack(os);
 
         }
         cout<<"Train data additive_share receiving ended!"<<endl;
 
         os.clear();
         for(int i=0;i<num_test_data;++i)
         {
             for(int j=0;j<num_features;++j)
             {   
                 random_data.randomize(prng);
                 m_test_additive_share_vec[i][j]=Z2<K>(m_test[i]->features[j])-random_data;
                 random_data.pack(os);
             }
         }
 
         m_player->send(os);
         cout<<"Test data additive_share sending ended!"<<endl;
 
     } 
     
 }
 
 
 void KNN_party_base::additive_share_data_vec(vector<Z2<K>>&shares,vector<Z2<K>>data_vec)
 {
     assert(data_vec.size()!=0&& data_vec.size()==shares.size() );
     octetStream os;
     PRNG prng;
     prng.ReSeed();
     for(int i=0;i<(int)data_vec.size();i++)
     {
         shares[i].randomize(prng);
         shares[i].pack(os);
         shares[i] = data_vec[i]-shares[i];
     }
     m_player->send(os);
 }
 
 void KNN_party_base::additive_share_data_vec(vector<Z2<K>>&shares)
 {
     octetStream os;
     m_player->receive(os);
     int size_of_share=shares.size();
     for(int i=0;i<size_of_share;i++)
         shares[i].unpack(os);
 }
 
 
 void parse_argv(int argc, const char** argv)
 {
   opt.add(
           "5000", // Default.
           0, // Required?
           1, // Number of args expected.
           0, // Delimiter if expecting multiple args.
           "Port number base to attempt to start connections from (default: 5000)", // Help description.
           "-pn", // Flag token.
           "--portnumbase" // Flag token.
   );
   opt.add(
           "", // Default.
           0, // Required?
           1, // Number of args expected.
           0, // Delimiter if expecting multiple args.
           "This m_player's number (required if not given before program name)", // Help description.
           "-p", // Flag token.
           "--m_player" // Flag token.
   );
   opt.add(
           "", // Default.
           0, // Required?
           1, // Number of args expected.
           0, // Delimiter if expecting multiple args.
           "Port to listen on (default: port number base + m_player number)", // Help description.
           "-mp", // Flag token.
           "--my-port" // Flag token.
   );
   opt.add(
           "localhost", // Default.
           0, // Required?
           1, // Number of args expected.
           0, // Delimiter if expecting multiple args.
           "Host where Server.x or party 0 is running to coordinate startup "
           "(default: localhost). "
           "Ignored if --ip-file-name is used.", // Help description.
           "-h", // Flag token.
           "--hostname" // Flag token.
   );
   opt.add(
           "", // Default.
           0, // Required?
           1, // Number of args expected.
           0, // Delimiter if expecting multiple args.
           "Filename containing list of party ip addresses. Alternative to --hostname and running Server.x for startup coordination.", // Help description.
           "-ip", // Flag token.
           "--ip-file-name" // Flag token.
   );
   opt.parse(argc, argv);
   if (opt.isSet("-p"))
     opt.get("-p")->getInt(playerno);
   else
     sscanf(argv[1], "%d", &playerno);
 }
 
 
 
 void gen_fake_dcf(int beta, int n)
 {
    // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
     int lambda_bytes = 16;
     PRNG prng;
     prng.InitSeed();
     fstream k0, k1, r0, r1, r2;
     k0.open("Player-Data/2-fss/k0", ios::out);
     k1.open("Player-Data/2-fss/k1", ios::out);
     r0.open("Player-Data/2-fss/r0", ios::out);
     r1.open("Player-Data/2-fss/r1", ios::out);
     r2.open("Player-Data/2-fss/r2", ios::out);
     octet seed[2][lambda_bytes];    
     bigint s[2][2], v[2][2],  t[2][2], tmp_t[2], convert[2], tcw[2], a, scw, vcw, va, tmp, tmp1, tmp_out;
     prng.InitSeed();
     prng.get(tmp, n);
     bytesFromBigint(&seed[0][0], tmp, lambda_bytes);
     k0 << tmp << " ";
     prng.get(tmp1, n);
     bytesFromBigint(&seed[1][0], tmp1, lambda_bytes);
     k1 << tmp1 << " ";
     prng.get(a, n);
     prng.get(tmp, n);
     r1 << a - tmp << " ";
     r0 << tmp << " ";
     r2 << tmp << " ";
     r0.close();
     r1.close();
     r2.close();
     tmp_t[0] = 0;
     tmp_t[1] = 1;
     int keep, lose;
     va = 0;
     //We can optimize keep into one bit here
     // generate the correlated word!
     for(int i = 0; i < n - 1; i++){
         keep = bigint(a >>( n - i - 1)).get_ui() & 1;
         lose = 1^keep;
         for(int j = 0; j < 2; j++){     
             prng.SetSeed(seed[j]);
             // k is used for left and right
             for(int k = 0; k < 2; k++){
                 prng.get(t[k][j], 1);
                 prng.get(v[k][j], n);
                 prng.get(s[k][j] ,n);
             }
         }
         scw = s[lose][0] ^ s[lose][1]; 
         // save convert(v0_lose) into convert[0]
         bytesFromBigint(&seed[0][0], v[lose][0], lambda_bytes);
         prng.SetSeed(seed[0]);
         prng.get(convert[0], n);     
         // save convert(v1_lose) into convert[1]
         bytesFromBigint(&seed[0][0], v[lose][1], lambda_bytes);
         prng.SetSeed(seed[0]);
         prng.get(convert[1], n);
         if(tmp_t[1])
             vcw = convert[0] + va - convert[1];
         else
             vcw = convert[1] - convert[0] - va;
         //keep == 1, lose = 0，so lose = LEFT
         if(keep)
             vcw = vcw + tmp_t[1]*(-beta) + (1-tmp_t[1]) * beta;
         // save convert(v0_keep) into convert[0]
         bytesFromBigint(&seed[0][0], v[keep][0], lambda_bytes);
         prng.SetSeed(seed[0]);
         prng.get(convert[0], n);
         // save convert(v1_keep) into convert[1]
         bytesFromBigint(&seed[0][0], v[keep][1], lambda_bytes);
         prng.SetSeed(seed[0]);
         prng.get(convert[1], n);
         va = va - convert[1] + convert[0] + tmp_t[1] * (-vcw) + (1-tmp_t[1]) * vcw;
         tcw[0] = t[0][0] ^ t[0][1] ^ keep ^ 1;
         tcw[1] = t[1][0] ^ t[1][1] ^ keep;
         k0 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
         k1 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
         bytesFromBigint(&seed[0][0],  s[keep][0] ^ (tmp_t[0] * scw), lambda_bytes);
         bytesFromBigint(&seed[1][0],  s[keep][1] ^ (tmp_t[1] * scw), lambda_bytes);
         bigintFromBytes(tmp_out, &seed[0][0], lambda_bytes);
         bigintFromBytes(tmp_out, &seed[1][0], lambda_bytes);
         tmp_t[0] = t[keep][0] ^ (tmp_t[0] * tcw[keep]);
         tmp_t[1] = t[keep][1] ^ (tmp_t[1] * tcw[keep]);
     }
     
     prng.SetSeed(seed[0]);
     prng.get(convert[0], n);
     prng.SetSeed(seed[1]);
     prng.get(convert[1], n);
     k0 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
     k1 << tmp_t[1]*(-1*(convert[1] - convert[0] - va)) + (1-tmp_t[1])*(convert[1] - convert[0] - va) << " ";
     k0.close();
     k1.close();
 
     return;
 }
 
 bigint evaluate(Z2<K> x, int n,int playerID)
 {
     call_evaluate_time++;
     auto start = std::chrono::high_resolution_clock::now();
 
     fstream k_in;
     PRNG prng;
     int b = playerID, xi;
     // Here represents the bytes that bigint will consume, the default number is 16, if the MAX_N_BITS is bigger than 128, then we should change.
     int lambda_bytes = 16;
     k_in.open("Player-Data/2-fss/k" + to_string(playerID), ios::in);
     octet seed[lambda_bytes], tmp_seed[lambda_bytes];
     // r is the random value generate by GEN
      static bigint s_hat[2], v_hat[2], t_hat[2], s[2], v[2], t[2], scw, vcw, tcw[2], convert[2], cw, tmp_t, tmp_v, tmp_out, term;
     k_in >> tmp_t;
     bytesFromBigint(&seed[0], tmp_t, lambda_bytes);
     tmp_t = b;
     tmp_v = 0;
     for(short i = 0; i < n - 1; ++i){
         xi = x.get_bit(n - i - 1);
         bigintFromBytes(tmp_out, &seed[0], lambda_bytes);
         k_in >> scw >> vcw >> tcw[0] >> tcw[1];
         prng.SetSeed(seed);
         for(short j = 0; j < 2; ++j){
             prng.get(t_hat[j], 1);
             prng.get(v_hat[j], n);
             prng.get(s_hat[j] ,n);
             s[j] = s_hat[j] ^ (tmp_t * scw);
             t[j] = t_hat[j] ^ (tmp_t * tcw[j]);
         }  
         bytesFromBigint(&tmp_seed[0], v_hat[0], lambda_bytes);
         prng.SetSeed(tmp_seed);
         prng.get(convert[0], n); 
         bytesFromBigint(&tmp_seed[0], v_hat[1], lambda_bytes);
         prng.SetSeed(tmp_seed);
         prng.get(convert[1], n);
         term = convert[xi] + tmp_t * vcw;
         tmp_v = tmp_v + b * (-1) * term + (1^b) * term;
         bytesFromBigint(&seed[0], s[xi], lambda_bytes);
         tmp_t = t[xi];
     }
     k_in >> cw;
     k_in.close();
     prng.SetSeed(seed);
     prng.get(convert[0], n);
     term = convert[0] + tmp_t * cw;
     tmp_v = tmp_v + b * (-1) * term + (1^b) * term;
 
     auto end = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> duration = end - start;
 
     // Accumulate the runtime of this execution into the global variable
     total_duration += duration;
 
     return tmp_v;  
 }
 
 
 void KNN_party_base::LabelCompute_test()
 {
     vector<Z2<K>>test_k_neighbors={Z2<K>(1),Z2<K>(0),Z2<K>(2),Z2<K>(2),Z2<K>(2)};
 
     vector<additive_share>share_k_neighbors(5);//k=5
 
     octetStream os;
     
     if(playerno==0)
     {
         PRNG prng;
         prng.ReSeed();
         for(int i=0;i<(int)test_k_neighbors.size();i++)
         {
             share_k_neighbors[i].randomize(prng);
             share_k_neighbors[i].pack(os);
             share_k_neighbors[i] = test_k_neighbors[i]-share_k_neighbors[i];
         }
         m_player->send(os);
     }else{
         m_player->receive(os);
         for(int i=0;i<share_k_neighbors.size();i++){
             share_k_neighbors[i].unpack(os);
         }
     }
     
     for(int i=0;i<test_k_neighbors.size();i++){
         Z2<K>tmp=reveal_one_num_to(share_k_neighbors[i],1);
         if(playerno==1){
             std::cout<<tmp<<" == "<<test_k_neighbors[i]<<endl;
         }
     }
 
     
     vector<vector<Z2<K>>> cmp_2d_vec_res(k_const,vector<Z2<K>>(k_const));
     vector<int>cmp_vec_idx;
     for(int i=0;i<k_const;i++){
         for(int j=0;j<k_const;j++){
             cmp_vec_idx.push_back(i);
             cmp_vec_idx.push_back(j);
         }
     }
     vector<Z2<K>>cmp_res_vec(cmp_vec_idx.size());
     compare_in_vec(share_k_neighbors,cmp_vec_idx,cmp_res_vec,true);
     for(int i=0;i<cmp_res_vec.size()/2;i++){
         cmp_2d_vec_res[cmp_vec_idx[i<<1]][cmp_vec_idx[(i<<1)+1]]=Z2<K>(m_playerno)-cmp_res_vec[i<<1];
     }
 
     vector<Z2<K>>v1(k_const*k_const),v2(k_const*k_const),res(k_const*k_const);
     for(int i=0;i<k_const;i++){
         for(int j=0;j<k_const;j++){
             v1[i*k_const+j]=cmp_2d_vec_res[i][j];
             v2[i*k_const+j]=cmp_2d_vec_res[j][i];
         }
     }
     mul_vector_additive(v1,v2,res,false);
 
     vector<array<Z2<K>,2>>label_list_count_array(k_const);
     for(int i=0;i<k_const;i++){
         Z2<K>tmp(0);
         for(int j=0;j<k_const;j++){
             tmp+=res[i*k_const+j];
         }
         label_list_count_array[i]={tmp,share_k_neighbors[i]};
     }
 
     cout<<"Test results:"<<endl;
     for(int i=0;i<k_const;i++){
         Z2<K>tmp_0=reveal_one_num_to(label_list_count_array[i][0],1);// Occurrence frequency
         Z2<K>tmp_1=reveal_one_num_to(label_list_count_array[i][1],1);// Corresponding label
         if(playerno==1){
             std::cout<<tmp_0<<"  "<<tmp_1<<endl;
         }
     }
 
 }
 
 
 void test_Z2()
 {
     long long x=65;
     long long y=999;
     // long long z=18446744073709543838;
     Z2<64>a(x);
     Z2<64>b(y);
     // cin>>b;
     cout<<a<<endl;
     cout<<b<<endl;
      std::vector<Z2<64>> tt(10);
     for (int i = 0; i < 10; i++) tt[i] = Z2<64>(i); // Initialize only 10 elements
     SignedZ2<K>c=a-b;
     cout<<SignedZ2<K>(a)<<endl;
     cout<<c<<endl;
 
     vector<Z2<K>>tmp1_vec(10);
     vector<Z2<K>>tmp2_vec(10);
     tmp1_vec[0]=Z2<64>(244);
     tmp2_vec[0]=Z2<64>(1);
     tmp1_vec[1]=Z2<64>(0);
     tmp2_vec[1]=Z2<64>(1);
     tmp1_vec[2]=Z2<64>(24);
     tmp2_vec[2]=Z2<64>(24);
     // KNN_party_base Kbase;
     cout<<endl;
 
 }
 
 void KNN_party_base::test_cmp()
 {
     vector<Z2<K>>test_vec={Z2<K>(1),Z2<K>(23),Z2<K>(2),Z2<K>(2),Z2<K>(2)};
     vector<additive_share>share_test_k(5);//k=5
     octetStream os;
     if(playerno==0)
     {
         PRNG prng;
         prng.ReSeed();
         for(int i=0;i<(int)share_test_k.size();i++)
         {
             share_test_k[i].randomize(prng);
             share_test_k[i].pack(os);
             share_test_k[i] = test_vec[i]-share_test_k[i];
         }
         m_player->send(os);
     }else{
         m_player->receive(os);
         for(int i=0;i<share_test_k.size();i++){
             share_test_k[i].unpack(os);
         }
     }
     
     for(int i=0;i<test_vec.size();i++){
         Z2<K>tmp=reveal_one_num_to(share_test_k[i],1);
         if(playerno==1){
             std::cout<<tmp<<" == "<<test_vec[i]<<endl;
         }
     }
     Z2<K>res1=secure_compare(share_test_k[0],share_test_k[1],true);//x1>x2-->1   x1<x2-->0   x1==x2-->0
     Z2<K>res2=secure_compare(share_test_k[1],share_test_k[0],true);//x1>x2-->1   x1<x2-->0   x1==x2-->0
     Z2<K>res3=secure_compare(share_test_k[0],share_test_k[0],true);//x1>x2-->1   x1<x2-->0   x1==x2-->0
     cout<<"Comparison test: x1 > x2 --> 1, else --> 0"<<endl;
     Z2<K>tmp=reveal_one_num_to(res1,1);
     if(playerno==1){
         std::cout<<tmp<<" == "<<test_vec[0]<<">"<<test_vec[1]<<endl;
     }
 
     tmp=reveal_one_num_to(res2,1);
     if(playerno==1){
         std::cout<<tmp<<" == "<<test_vec[1]<<">"<<test_vec[0]<<endl;
     }
 
     tmp=reveal_one_num_to(res3,1);
     if(playerno==1){
         std::cout<<tmp<<" == "<<test_vec[0]<<">"<<test_vec[0]<<endl;
     }
     
 
 }
