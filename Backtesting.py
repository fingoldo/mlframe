import pandas as pd
import numpy as np

#@profile
def FullBacktest(base,all_dates,all_spreads,cStartCapital,period_span_in_days,recency,min_periods_before,min_assets_available,backtest_chunk_size,verbose=False):
    res=pd.DataFrame(columns=['date','fTotalProfit','fTotalTurnOver','fROI','fTotalComm','lNumBets','fMaxDDPercent','lMaxDDDuration','num_available_assets'])
    if verbose:
        outcomes=pd.DataFrame(index=np.arange(len(base.index.unique())*len(base['level_spread'].unique())),columns=base.reset_index().columns)
    else:
        outcomes=None
    cntr=0
    one_day=np.timedelta64(1, 'D');n_days=np.timedelta64(period_span_in_days * recency, 'D');
    for cur_idx in range(min_periods_before+1,len(all_dates)-backtest_chunk_size):
        ########################################################################################################################################################################################################################################
        #Micro-sim inits
        ########################################################################################################################################################################################################################################
        lNumBets = 0;
        fTotalComm = 0; fTotalProfit = 0; fTotalTurnOver = 0;
        fMaxDDPercent = 0; lMaxDDDuration = 0; lHighestCapitalBetIndex = 0
        fCurBalance = cStartCapital; fHighestBlance = fCurBalance; fLowestBlance = fCurBalance
        if verbose:
            print (str(all_dates[cur_idx])+" weekly chunk started")
        for per in range (backtest_chunk_size):
            cur_date=all_dates[cur_idx+per]
            #print(cur_date)
            next_base=base.loc[cur_date]
            available_assets = next_base['ticker'].unique()
            num_available_assets=len(available_assets)
            if num_available_assets>=min_assets_available:
                #print("Available assets for that date: "+str(available_assets))
                ########################################################################################################################################################################################################################################
                #Need to select what capital % to use for each of available_assets, and what Spread
                ########################################################################################################################################################################################################################################

                ########################################################################################################################################################################################################################################
                #1. random selection of Spread at each step, capital gets divided uniformly between all available assets/markets
                ########################################################################################################################################################################################################################################

                #funds_allocated=np.ones(num_available_assets)*cStartCapital/num_available_assets
                #spreads_to_use=np.random.choice(all_spreads,num_available_assets,replace=True)

                ########################################################################################################################################################################################################################################
                #2. using at each step of Spread which has on average worked best before:
                ########################################################################################################################################################################################################################################
                # 2.1) for that asset since early days till "now"

                hist_perf = base.loc[(cur_date - n_days):(cur_date - one_day)].groupby(['ticker', 'level_spread'])

                #hist_perf = base.loc[cur_date].groupby(['ticker', 'level_spread'])  #Cheating!

                hist_perf_med=hist_perf['perf'].median().reset_index().set_index(['ticker'],inplace=False)
                spreads_to_use=[]

                best_historical_perfs_by_asset=[]
                for next_asset in range(num_available_assets):
                    fSpreadToUse=0;cur_best_perf=0
                    if available_assets[next_asset] in hist_perf_med.index:
                        possible_spreads_to_use=hist_perf_med.loc[available_assets[next_asset]]
                        if possible_spreads_to_use.size>0:
                            perfs=possible_spreads_to_use['perf'].values
                            the_ind = np.argmax(perfs)
                            best_historical_perf=perfs[the_ind]
                            if best_historical_perf>0:
                                best_spread_to_use=possible_spreads_to_use.iloc[[the_ind]]
                                cur_best_perf=best_spread_to_use['perf'].iloc[0]    #!!!
                                if verbose:
                                    print("Spread chosen for " + str(available_assets[next_asset]) + ":")
                                    display(best_spread_to_use)
                                    print("Its expected performance: "+str(best_spread_to_use['perf']))
                                fSpreadToUse=best_spread_to_use['level_spread'].iloc[0]
                    best_historical_perfs_by_asset.append(cur_best_perf)    #!!!
                    spreads_to_use.append(fSpreadToUse)

                funds_allocated=np.ones(num_available_assets)*cStartCapital/num_available_assets
                #print(best_historical_perfs_by_asset)
                avg_perf=np.median(best_historical_perfs_by_asset)
                funds_allocated[best_historical_perfs_by_asset>avg_perf]+=cStartCapital/num_available_assets/2
                funds_allocated[best_historical_perfs_by_asset<avg_perf]-=cStartCapital/num_available_assets/2
                #print (np.sum(funds_allocated))

                if verbose:
                    print("funds_allocated:"+str(funds_allocated))
                    print("spreads_to_use:"+str(spreads_to_use))

                ########################################################################################################################################################################################################################################
                #Capital calculation at the end of this week
                ########################################################################################################################################################################################################################################
                fProfit=0;fComm=0;fTurnover=0;prev_spread=0;
                for next_asset in range(num_available_assets):
                    funds=funds_allocated[next_asset]
                    if ((funds>0) & (spreads_to_use[next_asset]>0)):

                        ########################################################################################################################################################################################################################################
                        #next_sim=next_base[(next_base['ticker'] ==available_assets[next_asset]) & (next_base['level_spread'] == spreads_to_use[next_asset])]

                        c_t=np.where(next_base['ticker'].values == available_assets[next_asset])
                        if prev_spread!=spreads_to_use[next_asset]:
                            c_l=np.where(next_base['level_spread'].values == spreads_to_use[next_asset])
                            prev_spread=spreads_to_use[next_asset]
                        next_sim=next_base.iloc[np.intersect1d(c_t,c_l)]
                        ########################################################################################################################################################################################################################################

                        pr=next_sim['total_profit'].iloc[0]
                        if pr<2:
                            if verbose:
                                outcomes.iloc[cntr]=next_sim.reset_index().iloc[0]
                                cntr+=1

                            fComm+=next_sim['total_commission'].iloc[0]*funds
                            lNumBets+=next_sim['num_bets'].iloc[0]
                            roi=next_sim['roi'].iloc[0]
                            fProfit+=pr*funds
                            if roi!=0:
                                fTurnover+=pr/roi*funds

                fTotalComm+=fComm
                fCurBalance+=fProfit
                fTotalProfit+=fProfit
                fTotalTurnOver+=fTurnover

                if (fCurBalance > fHighestBlance) | (per==(backtest_chunk_size-1)):
                    fCurDD = (fHighestBlance - fLowestBlance) / fHighestBlance
                    if fCurDD > fMaxDDPercent:
                        fMaxDDPercent = fCurDD
                    lCurDDDuration = per - lHighestCapitalBetIndex
                    if lCurDDDuration > lMaxDDDuration:
                        lMaxDDDuration = lCurDDDuration

                    lHighestCapitalBetIndex = per

                    fHighestBlance = fCurBalance; fLowestBlance = fCurBalance
                else:
                    if fCurBalance < fLowestBlance:
                        fLowestBlance = fCurBalance
        if fTotalTurnOver>0:
            fROI=fTotalProfit/fTotalTurnOver
            res.loc[len(res)] = [all_dates[cur_idx],fTotalProfit,fTotalTurnOver,fROI,fTotalComm,lNumBets,fMaxDDPercent,lMaxDDDuration,num_available_assets]
        if verbose:
            print ('Weekly results: fTotalProfit='+str(fTotalProfit)+',fMaxDDPercent='+str(fMaxDDPercent)+',lNumBets='+str(lNumBets)+',fROI='+str(fROI)+',assets: '+str(num_available_assets))
            if cntr>1000:
                return res,outcomes
    return res,outcomes