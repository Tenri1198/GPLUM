#pragma once

void showGplumVersion(std::string version)
{
    if ( PS::Comm::getRank() == 0 ) {
        version.resize(16,' ');
        
        std::cout << "                                           \n "
                  << "     __________________________________   \n"
                  << "     /                                  \\  \n"
                  << "    |    ____ ____  _    _   _ __  __    | \n"
                  << "    |   / ___|  _ \\| |  | | | |  \\/  |   | \n"
                  << "    |  | |  _| |_) | |  | | | | |\\/| |   | \n"
                  << "    |  | |_| |  __/| |__| |_| | |  | |   | \n"
                  << "    |   \\____|_|   |_____\\___/|_|  |_|   | \n"
                  << "    |                                    | \n"
                  << "    |  Global Planetary Simulation Code  | \n"
                  << "    |    with Mass-dependent Cut-off     | \n"
                  << "    |       Version " << version << "     | \n"
                  << "     \\__________________________________/  \n"
                  << "                                           \n"
                  << "     Licence: MIT (see, https://github.com/YotaIshigaki/GPLUM/blob/master/LICENSE) \n"
                  << "                                           \n"
                  << "     Copyright (C) 2020                    \n"
                  << "       Yota Ishigaki, Junko Kominmi, Junichiro Makino, \n"
                  << "       Masaki Fujimoto and Masaki Iwasawa              \n"
                  << "                                                       \n";
    }
}

template <class Tpsys>
void calcMeanMass(Tpsys & pp,
                  PS::F64 & m_mean,
                  PS::F64 & m_max,
                  PS::F64 & nei_mean)
{
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    const PS::S32 n_glb = pp.getNumberOfParticleGlobal();
    PS::F64 m_sum_loc = 0.;
    PS::F64 m_max_loc = 0.;
    PS::S32 nei_sum_loc = 0;
    
    for (PS::S32 i=0; i<n_loc; i++ ){
        m_sum_loc += pp[i].mass;
        if ( pp[i].mass > m_max_loc ) m_max_loc = pp[i].mass;
        nei_sum_loc += pp[i].neighbor;
    }
    m_mean = PS::Comm::getSum(m_sum_loc) / n_glb;
    m_max = PS::Comm::getMaxValue(m_max_loc);
    nei_mean = (PS::F64)PS::Comm::getSum(nei_sum_loc) / n_glb;
}

template <class Tpsys>
void makeSnap(Tpsys & pp,
              PS::F64 time_sys,
              Energy e_init,
              Energy e_now,
              const char * dir_name,
              const PS::S32 isnap,
              const PS::S64 id_next)
{
    FileHeader header(pp.getNumberOfParticleGlobal(), id_next, time_sys, e_init, e_now);
    char filename[256];
    sprintf(filename, "%s/snap%06d.dat", dir_name, isnap);
    pp.writeParticleAscii(filename, header);
}

template <class Tpsys>
void makeSnapTmp(Tpsys & pp,
                  PS::F64 time_sys,
                  Energy e_init,
                  Energy e_now,
                  const char * dir_name,
                  const PS::S64 id_next)
{
    FileHeader header(pp.getNumberOfParticleGlobal(), id_next, time_sys, e_init, e_now);
    char filename[256];
    sprintf(filename, "%s/snap_tmp.dat", dir_name);
    pp.writeParticleBinary(filename, header);
}

template <class Tpsys>
void outputStep(Tpsys & pp,
                PS::F64 time_sys,
                Energy e_init,
                Energy e_now,
                PS::F64 de,
                PS::S32 n_col_tot,
                PS::S32 n_frag_tot,
                const char * dir_name,
                const PS::S32 isnap,
                const PS::S64 id_next,
                std::ofstream & fout_eng,
                Wtime wtime,
                PS::S32 n_largestcluster,
                PS::S32 n_cluster,
                PS::S32 n_isoparticle,
                bool bSnap=true)
{
    const PS::S32 n_tot = pp.getNumberOfParticleGlobal();

    if ( bSnap ) makeSnap(pp, time_sys, e_init, e_now, dir_name, isnap, id_next);

#ifdef OUTPUT_DETAIL
    PS::F64 m_mean = 0.;
    PS::F64 m_max = 0.;
    PS::F64 nei_mean = 0.;
    calcMeanMass(pp, m_mean, m_max, nei_mean);
#endif

    if(PS::Comm::getRank() == 0 && bSnap){
        //PS::F64 de =  e_now.calcEnergyError(e_init);
        //PS::F64 de_tmp = sqrt(de*de);
        //if( de_tmp > de_max ) de_max = de_tmp;
        fout_eng  << std::fixed<<std::setprecision(8)
                  << time_sys << "\t" << n_tot << "\t"
                  << std::scientific<<std::setprecision(15)
                  << e_now.etot << "\t" << de << "\t"
                  << n_largestcluster << "\t" << n_cluster << "\t" << n_isoparticle
#ifdef OUTPUT_DETAIL
                  << "\t" << m_max << "\t" << m_mean << "\t" << nei_mean
#endif
#ifdef CALC_WTIME
                  << "\t" << wtime.soft_step << "\t" << wtime.hard_step << "\t"
                  << wtime.calc_soft_force_step << "\t" << wtime.neighbor_search_step << "\t"
                  << wtime.calc_hard_force_step << "\t"
                  << wtime.create_cluster_step << "\t" << wtime.communication_step << "\t"
                  << wtime.output_step 
#endif
                  << std::endl;
    }
}

template <class Tpsys>
void setIDLocalAndMyrank(Tpsys & pp,
                         NeighborList & NList)
{
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    PS::S32 myrank = PS::Comm::getRank();
#pragma omp parallel for
    for(PS::S32 i=0; i<n_loc; i++){
        pp[i].id_local = i;
        pp[i].myrank = myrank;
        pp[i].inDomain = true;
        pp[i].isSent = false;
    }

    NList.makeIdMap(pp);
}

#ifdef USE_POLAR_COORDINATE
template <class Tpsys>
void setPosPolar(Tpsys & pp)
{
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    for(PS::S32 i=0; i<n_loc; i++) pp[i].setPosPolar();
}
#endif
/************************ここから修正しました*********************************/
template <class Tpsys>
PS::S32 particleCrossingOMF(Tpsys & pp,
                   PS::F64 & edisp,
                   PS::F64 & edisp_d)
{   
    PS::S32 mass_flag_loc = 0;
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    PS::S32 mass_flag_glb=0;
    PS::F64 edisp_loc = 0.;
    PS::F64 edisp_d_loc = 0.;
    PS::F64 mass_temp[n_loc];   //質量が変更される前の粒子質量を格納する配列
    PS::S32 pp_id[n_loc];        //質量が変更される前のIDを格納する配列
    PS::F64 delta_mass[n_loc];   //質量変化の差分を格納する配列
    const PS::F64 AU = 14959787070000.0; //[cm/AU]
    const PS::F64 sun_mass = 1.9884e33; //[g]
   
    for(PS::S32 i=0; i<n_loc; i++)  //質量が変更される前の粒子質量を格納する
    {
        mass_temp[i] = pp[i].mass;
        pp_id[i]  = pp[i].id;
    }

#pragma omp parallel for  //disk.hで元々行っていたOMF通過による粒子質量の変更に関する処理をこちらへ持ってきた
    for(PS::S32 i=0; i<n_loc; i++){
        PS::F64 r2 = pp[i].pos.x*pp[i].pos.x + pp[i].pos.y*pp[i].pos.y;
        PS::F64 r_inv = 1./sqrt(r2);
        PS::F64 r = r2 * r_inv;
        PS::F64 rho_dust = 1.0/sun_mass*AU*AU*AU; //ダスト密度(微惑星まで成長した場合)
        if(r<7.0 && pp[i].flag_gd==1) //OMFを通過した粒子への処理
        {
            mass_flag_loc = 1;
            std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h by crossing OMF(before) id:"<<pp[i].id<<" distance:"<<r<<" mass:"<<pp[i].mass<<" size:"<<pp[i].r_planet<<" flag:"<<pp[i].flag_gd<<std::endl;
		    pp[i].flag_gd=0;
		    pp[i].mass = pp[i].mass*1.844028699792144e-09/5.028e-20;
		    pp[i].r_planet = pow((0.75*pp[i].mass/(rho_dust*M_PI)),1./3.);
		    pp[i].f = 1.0;
		    pp[i].acc_gd=0.;
            std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h by crossing OMF(after) id:"<<pp[i].id<<" distance:"<<r<<" mass:"<<pp[i].mass<<" size:"<<pp[i].r_planet<<" flag:"<<pp[i].flag_gd<<std::endl;
	    }
    }
    mass_flag_glb = PS::Comm::getSum(mass_flag_loc);
    
    for(PS::S32 i=0; i<n_loc; i++)//粒子質量の変更に関するエネルギー処理をここに書く
    {   
        delta_mass[i] = (pp[i].mass-mass_temp[i]);
        
        if(delta_mass[i])  //質量が増えたわけだから、edisp_locには足していく必要がある(はず)
        {
            std::cout<<std::scientific<<"id:"<<pp[i].id<<" delta mass:"<<delta_mass[i]<<" mass:"<<pp[i].mass<<" temp_mass:"<<mass_temp[i]<<std::endl;
            std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h by crossing OMF(Energy change) id:"<<pp[i].id<<" mass:"<<pp[i].mass<<" mass:"<<mass_temp[i]<<" index:"<<pp_id[i]<<std::endl;
            edisp_loc += 0.5 * delta_mass[i] * pp[i].vel * pp[i].vel;
            edisp_d_loc += 0.5 * delta_mass[i] * pp[i].vel * pp[i].vel;
		    edisp_loc += delta_mass[i] * pp[i].phi_s;
            edisp_d_loc += delta_mass[i] * pp[i].phi_s;
		    edisp_loc += delta_mass[i] * pp[i].phi_d;
            edisp_d_loc += delta_mass[i] * pp[i].phi_d;
		    edisp_loc += delta_mass[i] * pp[i].phi;
        }
        for(PS::S32 j=0; j<i; j++)
        {
	        PS::F64 massi = delta_mass[i];
	        PS::F64 massj = delta_mass[j];
	        PS::F64vec posi = pp[i].pos;
		    PS::F64vec posj = pp[j].pos;
		    PS::F64vec dr = posi - posj;
            PS::F64 eps2 = FP_t::eps2;
            PS::F64 dr2 = dr*dr+eps2;
		    PS::F64 rinv = 1./sqrt(dr2);
		    edisp_loc -= massi * massj * rinv * (1.-cutoff_W2(dr2, pp[i].r_out_inv, pp[j].r_out_inv));
            edisp_d_loc -= massi * massj * rinv * (1.-cutoff_W2(dr2, pp[i].r_out_inv, pp[j].r_out_inv));
		}
    }

    edisp += PS::Comm::getSum(edisp_loc);
    edisp_d += PS::Comm::getSum(edisp_d_loc);
    
    return mass_flag_glb;
}

template <class Tpsys>
PS::S32 MergeParticle(Tpsys & pp,
                   PS::S32 n_col,
                   PS::F64 & edisp,
                   PS::F64 & edisp_d)
{   
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    const PS::S32 n_proc = PS::Comm::getNumberOfProc();
    PS::S32 n_remove = 0;
    PS::S32 * remove = new PS::S32[n_col];
    PS::F64 edisp_loc = 0.;
    PS::F64 edisp_d_loc = 0.;
    PS::F64 mass_temp[n_loc];
    PS::F64 delta_mass[n_loc];
    PS::S32 mass_flag_loc = 0;
    PS::S32 mass_flag_glb = 0;
    static std::vector<PS::S32> n_col_list;
    static std::vector<PS::S32> n_col_adr;
    static std::vector<FP_t> col_list_loc;
    static std::vector<FP_t> col_list_glb;
    n_col_list.resize(n_proc);
    n_col_adr.resize(n_proc);

    static std::vector<PS::S32> col_list;
    col_list.clear();

    for(PS::S32 i=0; i<n_loc; i++)  //質量が変更される前の粒子質量を格納する
    {
        mass_temp[i] = pp[i].mass;
    }

#pragma omp parallel for reduction (-:edisp_loc)  //粒子の衝突合体処理
    for ( PS::S32 i=0; i<n_loc; i++ ){
        bool flag_merge = 1;
        bool peble_proto_col = 0;
        if ( pp[i].isMerged && !pp[i].isDead) {
            for ( PS::S32 j=0; j<n_loc; j++ ){              
                if ( pp[j].id == pp[i].id && i != j ){
                    PS::F64vec vrel = pp[j].vel - pp[i].vel;
                    //std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h before [target:impactor]"<<pp[i].id<<":"<<pp[i].flag_gd<<" "<<pp[i].mass<<" "<<pp[j].id<<":"<<pp[j].flag_gd<<" "<<pp[j].mass<<std::endl;
                    if(pp[i].flag_gd==0 && pp[j].flag_gd==1)
                    {  
                        pp[j].mass = mass_temp[j]*1.844028699792144e-09/5.028e-20; 
                        //pp[i].mass += mj;
                        mass_flag_loc = 1;
                        peble_proto_col = 1;
                        /*
                        PS::F64 massd = pp[j].mass - mass_temp[j];
                        edisp_loc += 0.5 * massd * pp[j].vel * pp[j].vel;
		                edisp_loc += massd * pp[j].phi_s;
		                edisp_loc += massd * pp[j].phi_d;
		                edisp_loc += massd* pp[j].phi;
                        for(PS::S32 k=0; k<j; k++)
                        {   
                            delta_mass[k]=pp[k].mass - mass_temp[k];
                            delta_mass[j]=pp[j].mass - mass_temp[j];
	                        PS::F64 massk = delta_mass[k];
	                        PS::F64 massj = delta_mass[j];
	                        PS::F64vec posk = pp[k].pos;
		                    PS::F64vec posj = pp[j].pos;
		                    PS::F64vec dr = posk - posj;
                            PS::F64 eps2 = FP_t::eps2;
		                    PS::F64 rinv = 1./sqrt(dr*dr+eps2);
		                    edisp_loc +=  massk * massj * rinv;
		                }
                        */
                    }
                    else if(pp[i].flag_gd==1 && pp[j].flag_gd==0 && flag_merge == 1)
                    {  
                        pp[i].mass = mass_temp[i]*1.844028699792144e-09/5.028e-20;
                        //pp[i].mass += mj;
                        mass_flag_loc = 1;
                        flag_merge = 0;
                        peble_proto_col = 1;
                        /*
                        PS::F64 massd = pp[i].mass - mass_temp[i];
                        edisp_loc += 0.5 * massd * pp[i].vel * pp[i].vel;
		                edisp_loc += massd * pp[i].phi_s;
		                edisp_loc += massd * pp[i].phi_d;
		                edisp_loc += massd* pp[i].phi;
                        for(PS::S32 k=0; k<i; k++)
                        {
                            delta_mass[i]=pp[i].mass - mass_temp[i];
                            delta_mass[k]=pp[k].mass - mass_temp[k];
	                        PS::F64 massi = delta_mass[i];
	                        PS::F64 massk = delta_mass[k];
	                        PS::F64vec posi = pp[i].pos;
		                    PS::F64vec posk = pp[k].pos;
		                    PS::F64vec dr = posi - posk;
                            PS::F64 eps2 = FP_t::eps2;
		                    PS::F64 rinv = 1./sqrt(dr*dr+eps2);
		                    edisp_loc +=  massi * massk * rinv;
		                }*/
                    }
                    
                    pp[i].mass += pp[j].mass;
                    pp[i].vel = ( pp[i].mass*pp[i].vel + pp[j].mass*pp[j].vel )/(pp[i].mass+pp[j].mass);
		            pp[i].flag_gd &= pp[j].flag_gd;
		            pp[j].flag_gd = pp[i].flag_gd;
		            std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h after [target:impactor]"<<pp[i].id<<":"<<pp[i].flag_gd<<" "<<pp[i].mass<<" "<<pp[j].id<<":"<<pp[j].flag_gd<<" "<<pp[j].mass<<std::endl;
                    //pp[i].acc = ( mi*pp[i].acc + mj*pp[j].acc )/(mi+mj);
                    mass_temp[i]+=mass_temp[j];
#ifdef GAS_DRAG
                    pp[i].acc_gd = ( pp[i].mass*pp[i].acc_gd + pp[j].mass*pp[j].acc_gd )/(pp[i].mass+pp[j].mass);
#endif
                    pp[i].phi   = ( pp[i].mass*pp[i].phi   + pp[j].mass*pp[j].phi   )/(pp[i].mass+pp[j].mass);
                    pp[i].phi_d = ( pp[i].mass*pp[i].phi_d + pp[j].mass*pp[j].phi_d )/(pp[i].mass+pp[j].mass);
                    if(peble_proto_col)
                    {
                        //j粒子のremoveに対する処理
	                    edisp_loc -= 0.5 * pp[j].mass * pp[j].vel * pp[j].vel;
                        edisp_d_loc -= 0.5 * pp[j].mass * pp[j].vel * pp[j].vel;
		                edisp_loc -= pp[j].mass * pp[j].phi_s;
                        edisp_d_loc -= pp[j].mass * pp[j].phi_s;
		                edisp_loc -= pp[j].mass * pp[j].phi_d;
                        edisp_d_loc -= pp[j].mass * pp[i].phi_d;
		                edisp_loc -= pp[j].mass * pp[j].phi;
                    }
                    else
                    {
                        edisp_loc -= 0.5 * pp[i].mass*pp[j].mass/(pp[i].mass+pp[j].mass) * vrel*vrel;  
                    }

#pragma omp critical
                    {
                        remove[n_remove] = j;
                        n_remove ++;
                    }
#pragma omp critical 
                    {
                        col_list.push_back(i);
                    }

                    assert ( pp[i].pos == pp[j].pos );
                    assert ( pp[j].isDead );

                }
                delta_mass[i]=pp[i].mass-mass_temp[i];
                if(delta_mass[i])  //質量を手で人工的に増やしたので、edisp_locには足していく必要がある(はず)
                {
                    edisp_loc += 0.5 * delta_mass[i] * pp[i].vel * pp[i].vel;
                    edisp_d_loc += 0.5 * delta_mass[i] * pp[i].vel * pp[i].vel;
		            edisp_loc += delta_mass[i] * pp[i].phi_s;
                    edisp_d_loc += delta_mass[i] * pp[i].phi_s;
		            edisp_loc += delta_mass[i] * pp[i].phi_d;
                    edisp_d_loc += delta_mass[i] * pp[i].phi_d;
		            edisp_loc += delta_mass[i] * pp[i].phi;
                }
                for(PS::S32 k=0; k<i; k++)
                {
	                PS::F64 massi = delta_mass[i];
	                PS::F64 massk = delta_mass[k];
	                PS::F64vec posi = pp[i].pos;
		            PS::F64vec posk = pp[k].pos;
		            PS::F64vec dr = posi - posk;
                    PS::F64 eps2 = FP_t::eps2;
                    PS::F64 dr2 = dr*dr+eps2;
		            PS::F64 rinv = 1./sqrt(dr2);
		            edisp_loc += massi * massk * rinv * (1.-cutoff_W2(dr2, pp[i].r_out_inv, pp[k].r_out_inv));
                    edisp_d_loc += massi * massk * rinv * (1.-cutoff_W2(dr2, pp[i].r_out_inv, pp[k].r_out_inv));
		        }
            }   
            pp[i].isMerged = false;   
        }
    }
    PS::S32 n_col_loc = col_list.size();
    PS::S32 n_col_glb = PS::Comm::getSum(n_col_loc);
    mass_flag_glb = PS::Comm::getSum(mass_flag_loc);
       /*if ( n_col_glb ){
        
        //PS::S32 * n_remove_list   = nullptr;
        //PS::S32 * n_remove_adr    = nullptr;
        //FP_t *  remove_list_loc = nullptr;
        //FP_t *  remove_list_glb = nullptr;
        
        if ( PS::Comm::getRank() == 0 ){
            //n_remove_list   = new PS::S32[n_proc];
            //n_remove_adr    = new PS::S32[n_proc];
            //remove_list_glb = new FP_t[n_remove_glb];
            col_list_glb.resize(n_col_glb);
        }
        //remove_list_loc = new FP_t[n_remove_loc];
        col_list_loc.resize(n_col_loc);
        
#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
        MPI_Gather(&n_col_loc, 1, PS::GetDataType(n_col_loc),
                   &n_col_list[0], 1, PS::GetDataType(n_col_list[0]),  0, MPI_COMM_WORLD);
#else
        n_col_list[0]  = n_col_loc;
#endif
        //PS::Comm::gather(&n_remove_loc, 1, n_remove_list);
        
        if ( PS::Comm::getRank() == 0 ){
            PS::S32 tmp_col = 0;
            for ( PS::S32 i=0; i<n_proc; i++ ){
                n_col_adr[i]  = tmp_col;
                tmp_col += n_col_list[i];
            }
            assert ( n_col_glb == tmp_col );
        }

        for ( PS::S32 i=0; i<n_col_loc; i++ ) {
            col_list_loc[i] = pp[col_list.at(i)];
        }

#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
        MPI_Gatherv(&col_list_loc[0], n_col_loc,PS::GetDataType(col_list_loc[0]),
                    &col_list_glb[0], &n_col_list[0], &n_col_adr[0], PS::GetDataType(col_list_glb[0]), 0, MPI_COMM_WORLD);
#else
        for(PS::S32 i=0; i<n_col_loc; i++) col_list_glb[i] = col_list_loc[i];
#endif
        //PS::Comm::gatherV(remove_list_loc, n_remove_loc, remove_list_glb, n_remove_list, n_remove_adr);
        
        if ( PS::Comm::getRank() == 0 ){
            for ( PS::S32 i=0; i<n_col_glb; i++ ) {
            
                PS::F64    delta_massi = col_list_glb[i].mass-mass_temp[col_list.at(i)];
                PS::F64vec veli  = col_list_glb[i].vel;
                edisp_loc += 0.5*delta_massi* veli*veli;
                edisp_loc += delta_massi * col_list_glb[i].phi_s;
                edisp_loc += delta_massi * col_list_glb[i].phi_d;
                edisp_loc += delta_massi * col_list_glb[i].phi;
            
                for ( PS::S32 j=0; j<i; j++ ) {
                    if ( col_list_glb[i].id != col_list_glb[j].id ) {
                        PS::F64    delta_massj = col_list_glb[j].mass-mass_temp[col_list.at(j)];
                        PS::F64vec posi  = col_list_glb[i].pos;
                        PS::F64vec posj  = col_list_glb[j].pos;
                        PS::F64    eps2   = FP_t::eps2;
                        
                        PS::F64vec dr = posi - posj;
                        PS::F64    rinv = 1./sqrt(dr*dr + eps2);
                        
                        edisp_loc += delta_massi * delta_massj * rinv;
                    }
                }
        
                std::cerr << "Collision Particle " << col_list_glb[i].id << std::endl
                          << "Position : " << std::setprecision(15) << col_list_glb[i].pos << std::endl;
                fout_col2 << std::fixed<<std::setprecision(8)
                         << col_list_glb[i].time << "\t" << col_list_glb[i].id << "\t"
                         << std::scientific << std::setprecision(15) << col_list_glb[i].mass << "\t"
                         << col_list_glb[i].pos.x << "\t" << col_list_glb[i].pos.y << "\t" << col_list_glb[i].pos.z << "\t"
                         << col_list_glb[i].vel.x << "\t" << col_list_glb[i].vel.y << "\t" << col_list_glb[i].vel.z
                         << std::endl;
            }
            
            //delete [] n_remove_list;
            //delete [] n_remove_adr;
            //delete [] remove_list_glb;
        
        }
        
        //delete [] remove_list_loc;

#ifdef INDIRECT_TERM
        e_ind_before = calcIndirectEnergy(pp);
#endif
    }
/*#pragma omp parallel for reduction (-:edisp_loc)  //粒子の衝突合体時の質量増加に伴うエネルギー処理
    for ( PS::S32 i=0; i<n_loc; i++ ){
        if ( pp[i].isMerged ) {
            for ( PS::S32 j=0; j<n_loc; j++ ){    
                if ( pp[j].id == pp[i].id && i != j ){
                    delta_mass[i]=pp[i].mass-(mass_temp[i]+mass_temp[j]);
                    if(delta_mass[i])  //質量が増えたわけだから、edisp_locには足していく必要がある(はず)
                    {
                        //std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h by crossing OMF(Energy change) id:"<<pp[i].id<<" mass:"<<pp[i].mass<<" mass:"<<mass_temp[i]<<" index:"<<pp_id[i]<<std::endl;
                        edisp_loc += 0.5 * delta_mass[i] * pp[i].vel * pp[i].vel;
		                edisp_loc += delta_mass[i] * pp[i].phi_s;
		                edisp_loc += delta_mass[i] * pp[i].phi_d;
		                edisp_loc += delta_mass[i] * pp[i].phi;
                    }
                    for(PS::S32 j=0; j<i; j++)
                    {
	                    PS::F64 massi = delta_mass[i];
	                    PS::F64 massj = delta_mass[j];
	                    PS::F64vec posi = pp[i].pos;
		                PS::F64vec posj = pp[j].pos;
		                PS::F64vec dr = posi - posj;
                        PS::F64 eps2 = FP_t::eps2;
		                PS::F64 rinv = 1./sqrt(dr*dr+eps2);
		                edisp_loc +=  massi * massj * rinv;
		            }
                }
            }
            pp[i].isMerged = false;   
        }
    }*/
    PS::Comm::barrier();
    edisp += PS::Comm::getSum(edisp_loc);
    edisp_d += PS::Comm::getSum(edisp_d_loc);
    if ( n_remove ){
        pp.removeParticle(remove, n_remove);
    }
    delete [] remove;
    //edisp += PS::Comm::getSum(edisp_loc);
    return mass_flag_glb;
}
/************************ここまで修正しました*********************************/
template <class Tpsys>
PS::S32 removeParticlesOutOfBoundary(Tpsys & pp,
                                     PS::F64 & edisp,
                                     const PS::F64 r_max,
                                     const PS::F64 r_min,
                                     std::ofstream & fout_rem)
{
    const PS::F64 rmax2 = r_max*r_max;
    const PS::F64 rmin2 = r_min*r_min;
    PS::F64 edisp_loc = 0.;
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    const PS::S32 n_proc = PS::Comm::getNumberOfProc();
    PS::S32 n_colrem_loc = 0;
    PS::S32 n_colrem_glb = 0;

    static std::vector<PS::S32> n_remove_list;
    static std::vector<PS::S32> n_remove_adr;
    static std::vector<FP_t> remove_list_loc;
    static std::vector<FP_t> remove_list_glb;
    n_remove_list.resize(n_proc);
    n_remove_adr.resize(n_proc);

    static std::vector<PS::S32> remove_list;
    remove_list.clear();

#ifdef INDIRECT_TERM
    PS::F64 e_ind_before = 0.;
    PS::F64 e_ind_after  = 0.;
#endif
    
#pragma omp parallel for
    for ( PS::S32 i=0; i<n_loc; i++ ){
        PS::F64vec posi = pp[i].pos;
        PS::F64    pos2 = posi*posi;
        if ( pos2 > rmax2 || pos2 < rmin2 ){
#pragma omp critical 
            {
                remove_list.push_back(i);
            }
        }
    }

    PS::S32 n_remove_loc = remove_list.size();
    PS::S32 n_remove_glb = PS::Comm::getSum(n_remove_loc);

    /*if ( n_remove_glb == 1 ){

        if ( n_remove_loc ) {
            PS::S32 i_remove = remove_list.at(0);
            
            PS::F64    massi = pp[i_remove].mass;
            PS::F64vec veli = pp[i_remove].vel;
            edisp_loc -= 0.5*massi* veli*veli;
            edisp_loc -= massi * pp[i_remove].phi_s;
            edisp_loc -= massi * pp[i_remove].phi_d;
            edisp_loc -= massi * pp[i_remove].phi;
        
            std::cerr << "Remove Particle " << pp[i_remove].id << std::endl
                      << "Position : " << std::setprecision(15) << pp[i_remove].pos << std::endl;
            fout_rem << std::fixed<<std::setprecision(8)
                     << pp[i_remove].time << "\t" << pp[i_remove].id << "\t"
                     << std::scientific << std::setprecision(15) << pp[i_remove].mass << "\t"
                     << pp[i_remove].pos.x << "\t" << pp[i_remove].pos.y << "\t" << pp[i_remove].pos.z << "\t"
                     << pp[i_remove].vel.x << "\t" << pp[i_remove].vel.y << "\t" << pp[i_remove].vel.z
                     << std::endl;
        }
        
        } else if ( n_remove_glb > 1 ){*/
    
    if ( n_remove_glb ){
        
        //PS::S32 * n_remove_list   = nullptr;
        //PS::S32 * n_remove_adr    = nullptr;
        //FP_t *  remove_list_loc = nullptr;
        //FP_t *  remove_list_glb = nullptr;
        
        if ( PS::Comm::getRank() == 0 ){
            //n_remove_list   = new PS::S32[n_proc];
            //n_remove_adr    = new PS::S32[n_proc];
            //remove_list_glb = new FP_t[n_remove_glb];
            remove_list_glb.resize(n_remove_glb);
        }
        //remove_list_loc = new FP_t[n_remove_loc];
        remove_list_loc.resize(n_remove_loc);
        
#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
        MPI_Gather(&n_remove_loc, 1, PS::GetDataType(n_remove_loc),
                   &n_remove_list[0], 1, PS::GetDataType(n_remove_list[0]),  0, MPI_COMM_WORLD);
#else
        n_remove_list[0]  = n_remove_loc;
#endif
        //PS::Comm::gather(&n_remove_loc, 1, n_remove_list);
        
        if ( PS::Comm::getRank() == 0 ){
            PS::S32 tmp_remove = 0;
            for ( PS::S32 i=0; i<n_proc; i++ ){
                n_remove_adr[i]  = tmp_remove;
                tmp_remove += n_remove_list[i];
            }
            assert ( n_remove_glb == tmp_remove );
        }

        for ( PS::S32 i=0; i<n_remove_loc; i++ ) {
            remove_list_loc[i] = pp[remove_list.at(i)];
        }

#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
        MPI_Gatherv(&remove_list_loc[0], n_remove_loc,                        PS::GetDataType(remove_list_loc[0]),
                    &remove_list_glb[0], &n_remove_list[0], &n_remove_adr[0], PS::GetDataType(remove_list_glb[0]), 0, MPI_COMM_WORLD);
#else
        for(PS::S32 i=0; i<n_remove_loc; i++) remove_list_glb[i] = remove_list_loc[i];
#endif
        //PS::Comm::gatherV(remove_list_loc, n_remove_loc, remove_list_glb, n_remove_list, n_remove_adr);
        
        if ( PS::Comm::getRank() == 0 ){
            for ( PS::S32 i=0; i<n_remove_glb; i++ ) {
            
                PS::F64    massi = remove_list_glb[i].mass;
                PS::F64vec veli  = remove_list_glb[i].vel;
                edisp_loc -= 0.5*massi* veli*veli;
                edisp_loc -= massi * remove_list_glb[i].phi_s;
                edisp_loc -= massi * remove_list_glb[i].phi_d;
                edisp_loc -= massi * remove_list_glb[i].phi;
            
                for ( PS::S32 j=0; j<i; j++ ) {
                    if ( remove_list_glb[i].id != remove_list_glb[j].id ) {
                        PS::F64    massj = remove_list_glb[j].mass;
                        PS::F64vec posi  = remove_list_glb[i].pos;
                        PS::F64vec posj  = remove_list_glb[j].pos;
                        PS::F64    eps2   = FP_t::eps2;
                        
                        PS::F64vec dr = posi - posj;
                        PS::F64    rinv = 1./sqrt(dr*dr + eps2);
                        
                        edisp_loc += - massi * massj * rinv;
                    }
                }
        
                std::cerr << "Remove Particle " << remove_list_glb[i].id << std::endl
                          << "Position : " << std::setprecision(15) << remove_list_glb[i].pos << std::endl;
                fout_rem << std::fixed<<std::setprecision(8)
                         << remove_list_glb[i].time << "\t" << remove_list_glb[i].id << "\t"
                         << std::scientific << std::setprecision(15) << remove_list_glb[i].mass << "\t"
                         << remove_list_glb[i].pos.x << "\t" << remove_list_glb[i].pos.y << "\t" << remove_list_glb[i].pos.z << "\t"
                         << remove_list_glb[i].vel.x << "\t" << remove_list_glb[i].vel.y << "\t" << remove_list_glb[i].vel.z
                         << std::endl;
            }
            
            //delete [] n_remove_list;
            //delete [] n_remove_adr;
            //delete [] remove_list_glb;
        
        }
        
        //delete [] remove_list_loc;

#ifdef INDIRECT_TERM
        e_ind_before = calcIndirectEnergy(pp);
#endif
    }
    
    if (n_remove_loc) pp.removeParticle(&remove_list[0], n_remove_loc);
            
    edisp += PS::Comm::getSum(edisp_loc);

#ifdef INDIRECT_TERM
    if (n_remove_glb) {
        e_ind_after = calcIndirectEnergy(pp);
        edisp += e_ind_after - e_ind_before;
    }
#endif

    return n_remove_glb;
}

template <class Tpsys>
void correctEnergyForGas(Tpsys & pp,
                         PS::F64 & edisp_gd,
                         bool second)
{// energy correction for gas drag
    PS::F64 edisp_gd_loc = 0.;
    PS::F64 coef = 0.25; if (second) coef *= -1.;
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    
#pragma omp parallel for reduction(+:edisp_gd_loc)
    for(PS::S32 i=0; i<n_loc; i++){
        edisp_gd_loc += pp[i].mass * pp[i].acc_gd
            * (pp[i].vel + coef * pp[i].acc_gd * FP_t::dt_tree);
    }
    edisp_gd += 0.5 * FP_t::dt_tree * PS::Comm::getSum(edisp_gd_loc);
}

template <class Tpsys>
PS::F64 calc_mass(Tpsys & pp)
{
    PS::F64 total_mass = 0.;
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    for(int i=0;i<n_loc;i++)
    {
        total_mass+=pp[i].mass;
    }
    return total_mass;
}


template <class Tpsys>
void collision_calc( Tpsys & pp,
		     PS::S32 n_col,
		     PS::F64 & edisp,
             PS::F64 & edisp_d)
{
    const PS::S32 n_loc = pp.getNumberOfParticleLocal();
    const PS::S32 n_proc = PS::Comm::getNumberOfProc();
    PS::F64 edisp_loc = 0.;
    PS::F64 edisp_d_loc = 0.;
    PS::S32 n_colrem_loc = 0;
    PS::S32 n_colrem_glb = 0;

    static std::vector<PS::S32> n_colrem_list;
    static std::vector<PS::S32> n_colrem_adr;
    static std::vector<FP_t> colrem_list_loc;
    static std::vector<FP_t> colrem_list_glb;
    n_colrem_list.resize(n_proc);
    n_colrem_adr.resize(n_proc);


    PS::S32 n_remove = 0;
    PS::S32 * remove = new PS::S32[n_col];
    PS::F64 mass_temp[n_loc];
    PS::F64 delta_mass[n_loc];
    PS::S32 mass_flag_loc = 0;
    PS::S32 mass_flag_glb = 0;

    static std::vector<PS::S32> colrem_list;
    colrem_list.clear();


    for(PS::S32 i=0; i<n_loc; i++)  //質量が変更される前の粒子質量を格納する
    {
        mass_temp[i] = pp[i].mass;
    }

#pragma omp parallel for reduction (-:edisp_loc)  //粒子の衝突合体処理
    for ( PS::S32 i=0; i<n_loc; i++ ){
        bool flag_merge = 1;
        bool peble_proto_col = 0;
        if ( pp[i].isMerged && !pp[i].isDead) {
            for ( PS::S32 j=0; j<n_loc; j++ ){              
                if ( pp[j].id == pp[i].id && i != j ){
                    PS::F64vec vrel = pp[j].vel - pp[i].vel;
                    //std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h before [target:impactor]"<<pp[i].id<<":"<<pp[i].flag_gd<<" "<<pp[i].mass<<" "<<pp[j].id<<":"<<pp[j].flag_gd<<" "<<pp[j].mass<<std::endl;
                    if(pp[i].flag_gd==0 && pp[j].flag_gd==1)
                    {  
                        pp[j].mass = mass_temp[j]*1.844028699792144e-09/5.028e-20; 
                        //pp[i].mass += mj;
                        mass_flag_loc = 1;
                        peble_proto_col = 1;
                    }
                    else if(pp[i].flag_gd==1 && pp[j].flag_gd==0 && flag_merge == 1)
                    {  
                        pp[i].mass = mass_temp[i]*1.844028699792144e-09/5.028e-20;
                        //pp[i].mass += mj;
                        mass_flag_loc = 1;
                        flag_merge = 0;
                        peble_proto_col = 1;
                    }
                    
                    pp[i].mass += pp[j].mass;
                    pp[i].vel = ( pp[i].mass*pp[i].vel + pp[j].mass*pp[j].vel )/(pp[i].mass+pp[j].mass);
		            pp[i].flag_gd &= pp[j].flag_gd;
		            pp[j].flag_gd = pp[i].flag_gd;
		            std::cout<<std::scientific<<std::setprecision(16)<<"flag_gd checker at func.h after [target:impactor]"<<pp[i].id<<":"<<pp[i].flag_gd<<" "<<pp[i].mass<<" "<<pp[j].id<<":"<<pp[j].flag_gd<<" "<<pp[j].mass<<std::endl;
                    //pp[i].acc = ( mi*pp[i].acc + mj*pp[j].acc )/(mi+mj);
                    mass_temp[i]+=mass_temp[j];
#ifdef GAS_DRAG
                    pp[i].acc_gd = ( pp[i].mass*pp[i].acc_gd + pp[j].mass*pp[j].acc_gd )/(pp[i].mass+pp[j].mass);
#endif
                    pp[i].phi   = ( pp[i].mass*pp[i].phi   + pp[j].mass*pp[j].phi   )/(pp[i].mass+pp[j].mass);
                    pp[i].phi_d = ( pp[i].mass*pp[i].phi_d + pp[j].mass*pp[j].phi_d )/(pp[i].mass+pp[j].mass);
                    edisp_loc -= 0.5 * pp[i].mass*pp[j].mass/(pp[i].mass+pp[j].mass) *vrel * vrel; 

#pragma omp critical
                    {
                        remove[n_remove] = j;
                        n_remove ++;
                    }
#pragma omp critical 
                    {
                        colrem_list.push_back(j);
                    }

                    assert ( pp[i].pos == pp[j].pos );
                    assert ( pp[j].isDead );
                     std::cout<<"L826"<<std::endl;
                }
                delta_mass[i]=pp[i].mass-mass_temp[i];
                if(delta_mass[i])  //質量を手で人工的に増やしたので、edisp_locには足していく必要がある(はず)
                {
                    edisp_loc += 0.5 * delta_mass[i] * pp[i].vel * pp[i].vel;
                    edisp_d_loc += 0.5 * delta_mass[i] * pp[i].vel * pp[i].vel;
		            edisp_loc += delta_mass[i] * pp[i].phi_s;
                    edisp_d_loc += delta_mass[i] * pp[i].phi_s;
		            edisp_loc += delta_mass[i] * pp[i].phi_d;
                    edisp_d_loc += delta_mass[i] * pp[i].phi_d;
		            edisp_loc += delta_mass[i] * pp[i].phi;
                    for(PS::S32 k=0; k<i; k++)
                    {
	                PS::F64 massi = delta_mass[i];
	                PS::F64 massk = delta_mass[k];
	                PS::F64vec posi = pp[i].pos;
		            PS::F64vec posk = pp[k].pos;
		            PS::F64vec dr = posi - posk;
                    PS::F64 eps2 = FP_t::eps2;
                    PS::F64 dr2 = dr*dr+eps2;
		            PS::F64 rinv = 1./sqrt(dr2);
		            edisp_loc += massi * massk * rinv * (1.-cutoff_W2(dr2, pp[i].r_out_inv, pp[k].r_out_inv));
                    edisp_d_loc += massi * massk * rinv * (1.-cutoff_W2(dr2, pp[i].r_out_inv, pp[k].r_out_inv));
		            }
                }
            }   
            pp[i].isMerged = false;   
        }
    }
    mass_flag_glb = PS::Comm::getSum(mass_flag_loc);
    std::cout<<"L857"<<std::endl;
    
    n_colrem_loc = colrem_list.size();
    n_colrem_glb = PS::Comm::getSum(n_colrem_loc);
    
    if( n_colrem_glb ){
        if(PS::Comm::getRank() == 0)
        {
            colrem_list_glb.resize(n_colrem_glb);
        }
        colrem_list_loc.resize(n_colrem_loc);
        std::cout<<"L868"<<std::endl;
#ifdef PARTICLE_SIMULATOR_PARALLEL
	     MPI_Gather(&n_remove_loc, 1, PS::GetDataType(n_remove_loc),
                    &n_remove_list[0], 1, PS::GetDataType(n_colrem_list[0]), 0, MPI_COMM_WORLD));
#else
         n_colrem_list[0] = n_colrem_loc;
#endif
	    if( PS::Comm::getRank() == 0 ){
	        PS::S32 tmp_colrem = 0;
	        for ( PS::S32 i=0; i<n_proc; i++)
            {
	            n_colrem_adr[i] = tmp_colrem;
                std::cout<<"# of proc"<<i<<":"<<"n_colrem_list"<<n_colrem_list[i]<<std::endl;
		        tmp_colrem += n_colrem_list[i];
	        }
            std::cout<<"n_colrem_glb:"<<n_colrem_glb<<" tmp_colrem:"<<tmp_colrem<<std::endl;
	        assert ( n_colrem_glb == tmp_colrem );
	    }

        for(PS::S32 i=0; i<n_colrem_loc; i++)
        {
	        colrem_list_loc[i] = pp[colrem_list.at(i)];
        }
        std::cout<<"L887"<<std::endl;
#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
	    MPI_Gatherv(&colrem_list_loc[0], n_colrem_loc,PS::GetDataType(colrem_list_loc[0]),
		    &colrem_list_glb[0], &n_colrem_list[0], &n_colrem_adr[0], PS::GetDataType(colrem_list_glb[0]), 0, MPI_COMM_WORLD);
#else
        for(PS::S32 i=0; i<n_colrem_loc; i++) colrem_list_glb[i] = colrem_list_loc[i];
#endif

        if( PS::Comm::getRank() == 0 ){
            for(PS::S32 i=0; i<n_colrem_glb; i++){

	            PS::F64 massi = colrem_list_glb[i].mass;
		        PS::F64vec veli = colrem_list_glb[i].vel;
	            edisp_loc -= 0.5 * massi * veli * veli;
	            edisp_loc -= massi * colrem_list_glb[i].phi_s;
	            edisp_loc -= massi * colrem_list_glb[i].phi_d;
	            edisp_loc -= massi * colrem_list_glb[i].phi;
	  
	            for(PS::S32 j=0; j<i; j++){
	                if(colrem_list_glb[j].id != colrem_list_glb[i].id){
		                PS::F64 massj = colrem_list_glb[j].mass;
		                PS::F64vec posi = colrem_list_glb[i].pos;
		                PS::F64vec posj = colrem_list_glb[j].pos;
		                PS::F64 eps2 = FP_t::eps2;

		                PS::F64vec dr = posi - posj;
		                PS::F64 rinv = 1./sqrt(dr*dr + eps2);

		                edisp_loc += -massi * massj * rinv;
	                }  	    
                }   
            }
        }
        std::cout<<"L920"<<std::endl;
#ifdef INDIRECT_TERM
	    e_ind_before = calcIndirectEnergy(pp);
#endif
    }

    if (n_colrem_loc) pp.removeParticle(&colrem_list[0], n_colrem_loc);
    std::cout<<"L927"<<std::endl;  
    edisp += PS::Comm::getSum(edisp_loc);
}

