#ifndef TABLE
#define TABLE
#define bit_mask(n) (1<<(n))

__constant__ int d_movable_piece_table[64][6][2];
__constant__ int d_step_table[2][25][4];

const int movable_piece_table[64][6][2]={
 {{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1}}
,{{ 0,-1},{ 0,-1},{ 0,-1},{ 0,-1},{ 0,-1},{ 0,-1}}
,{{ 1,-1},{ 1,-1},{ 1,-1},{ 1,-1},{ 1,-1},{ 1,-1}}
,{{ 0,-1},{ 1,-1},{ 1,-1},{ 1,-1},{ 1,-1},{ 1,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 2,-1},{ 2,-1},{ 2,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 2,-1},{ 2,-1},{ 2,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 2,-1},{ 2,-1},{ 2,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 2,-1},{ 2,-1},{ 2,-1}}
,{{ 3,-1},{ 3,-1},{ 3,-1},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 0,-1},{ 0, 3},{ 0, 3},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 1,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 0,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 3,-1},{ 3,-1}}
,{{ 4,-1},{ 4,-1},{ 4,-1},{ 4,-1},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 0, 4},{ 0, 4},{ 0, 4},{ 4,-1},{ 4,-1}}
,{{ 1,-1},{ 1,-1},{ 1, 4},{ 1, 4},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 1,-1},{ 1, 4},{ 1, 4},{ 4,-1},{ 4,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 2, 4},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 2, 4},{ 4,-1},{ 4,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 2, 4},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 2, 4},{ 4,-1},{ 4,-1}}
,{{ 3,-1},{ 3,-1},{ 3,-1},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 0, 3},{ 0, 3},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 1,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 4,-1},{ 4,-1}}
,{{ 5,-1},{ 5,-1},{ 5,-1},{ 5,-1},{ 5,-1},{ 5,-1}}
,{{ 0,-1},{ 0, 5},{ 0, 5},{ 0, 5},{ 0, 5},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 1, 5},{ 1, 5},{ 1, 5},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 1, 5},{ 1, 5},{ 1, 5},{ 5,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 2, 5},{ 2, 5},{ 5,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 2, 5},{ 2, 5},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 2, 5},{ 2, 5},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 2, 5},{ 2, 5},{ 5,-1}}
,{{ 3,-1},{ 3,-1},{ 3,-1},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 0,-1},{ 0, 3},{ 0, 3},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 3, 5},{ 5,-1}}
,{{ 4,-1},{ 4,-1},{ 4,-1},{ 4,-1},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 0, 4},{ 0, 4},{ 0, 4},{ 4,-1},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 1, 4},{ 1, 4},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 1, 4},{ 1, 4},{ 4,-1},{ 5,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 2, 4},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 2, 4},{ 4,-1},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 2, 4},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 2, 4},{ 4,-1},{ 5,-1}}
,{{ 3,-1},{ 3,-1},{ 3,-1},{ 3,-1},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 0, 3},{ 0, 3},{ 3,-1},{ 4,-1},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 1, 3},{ 3,-1},{ 4,-1},{ 5,-1}}
,{{ 2,-1},{ 2,-1},{ 2,-1},{ 3,-1},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 0, 2},{ 2,-1},{ 3,-1},{ 4,-1},{ 5,-1}}
,{{ 1,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 4,-1},{ 5,-1}}
,{{ 0,-1},{ 1,-1},{ 2,-1},{ 3,-1},{ 4,-1},{ 5,-1}}
};
// movable_piece1 = movable_piece_table[piece_bits[moving_color]][dice][0/1];

// format: {step1,step2,step3,length}
// [moving_color][piece1_pos][]     012: 3 direction, 3: move_count(how many choice you have)
const int step_table[2][25][4]=
{
    {
        { 1, 5, 6, 3},{ 2, 6, 7, 3},{ 3, 7, 8, 3},{ 4, 8, 9, 3},{ 9,-1,-1, 1},
        { 6,10,11, 3},{ 7,11,12, 3},{ 8,12,13, 3},{ 9,13,14, 3},{14,-1,-1, 1},
        {11,15,16, 3},{12,16,17, 3},{13,17,18, 3},{14,18,19, 3},{19,-1,-1, 1},
        {16,20,21, 3},{17,21,22, 3},{18,22,23, 3},{19,23,24, 3},{24,-1,-1, 1},
        {21,-1,-1, 1},{22,-1,-1, 1},{23,-1,-1, 1},{24,-1,-1, 1},{-1,-1,-1, 0}
    },
    {
        {-1,-1,-1, 0},{ 0,-1,-1, 1},{ 1,-1,-1, 1},{ 2,-1,-1, 1},{ 3,-1,-1, 1},
        { 0,-1,-1, 1},{ 5, 1, 0, 3},{ 6, 2, 1, 3},{ 7, 3, 2, 3},{ 8, 4, 3, 3},
        { 5,-1,-1, 1},{10, 6, 5, 3},{11, 7, 6, 3},{12, 8, 7, 3},{13, 9, 8, 3},
        {10,-1,-1, 1},{15,11,10, 3},{16,12,11, 3},{17,13,12, 3},{18,14,13, 3},
        {15,-1,-1, 1},{20,16,15, 3},{21,17,16, 3},{22,18,17, 3},{23,19,18, 3}
    }
};

// #define UINT128(high, low) ((__uint128_t(high) << 64) | __uint128_t(low))
// // zobrist table
// const __uint128_t zobrist[12][25] =
// {
//     {UINT128(1401874464333547133ULL, 12680320617958819676ULL), UINT128(8543685260210134095ULL, 5723210886821625140ULL), UINT128(11445716761529423746ULL, 6638394709650885341ULL), UINT128(14820266406887846083ULL, 14843007271225781569ULL), UINT128(7568220155852078611ULL, 1948408050018597165ULL), UINT128(5590830135204295569ULL, 13951228123923287032ULL), UINT128(12836676716682212222ULL, 11620224041008816253ULL), UINT128(16265342276460333228ULL, 10046310585380055238ULL), UINT128(16109367102750247134ULL, 501959109850507446ULL), UINT128(517823345688455450ULL, 4425343333022269113ULL), UINT128(10479067395846157177ULL, 6342302574108009087ULL), UINT128(13910292609877216589ULL, 1689049295929235913ULL), UINT128(13625175276617029692ULL, 12271842902987546540ULL), UINT128(466949590771302107ULL, 1644234598920806455ULL), UINT128(6477811250062420358ULL, 2689531957784857756ULL), UINT128(12152043776134712194ULL, 13774669804552460087ULL), UINT128(14528760230801936685ULL, 1492687113494526783ULL), UINT128(16772516204105805686ULL, 14790109902601057657ULL), UINT128(10503921038516537537ULL, 501795280331488817ULL), UINT128(18316919773911274224ULL, 10021316365004262902ULL), UINT128(14419623613706165714ULL, 7079726616456884805ULL), UINT128(13407548997040463067ULL, 10695967690149691597ULL), UINT128(325748231454970844ULL, 15637857696199619518ULL), UINT128(4825321484623204409ULL, 2037721873559823396ULL), UINT128(15447686159333764081ULL, 7369731577165721502ULL)},
//     {UINT128(7681779585188680094ULL, 5848996292442755984ULL), UINT128(9353493507507279223ULL, 7431639993340316287ULL), UINT128(15463455560000352775ULL, 8232041652053447914ULL), UINT128(14602598669890164319ULL, 1717668266780914580ULL), UINT128(3144197655138332943ULL, 2984944495220314322ULL), UINT128(13482561576499054676ULL, 16453209164441994947ULL), UINT128(13588454682514006403ULL, 5910226467665515362ULL), UINT128(882482547333519299ULL, 2823732876508397137ULL), UINT128(8980780420504456954ULL, 9174524111156965413ULL), UINT128(2698798777592451032ULL, 13760295846169639758ULL), UINT128(8048398194454811452ULL, 7857516028000602000ULL), UINT128(5792560339324325233ULL, 14258979435966703507ULL), UINT128(15944110096948304423ULL, 10841723488414423485ULL), UINT128(17249515710973563392ULL, 17567007324389496051ULL), UINT128(12308491550362880185ULL, 11847515402829848796ULL), UINT128(15748583356119793187ULL, 10569583836202708633ULL), UINT128(6495519827784063004ULL, 15080296316271047075ULL), UINT128(11829095887644072280ULL, 14769548190726660127ULL), UINT128(9148344557689882350ULL, 4236106682692791748ULL), UINT128(7437991928569007710ULL, 6222184594600879001ULL), UINT128(16993487295007929704ULL, 4260280915952731558ULL), UINT128(5396928249223409960ULL, 8942421424882742765ULL), UINT128(12855598771650675388ULL, 11014649117849897758ULL), UINT128(3583255294219385213ULL, 1311484385254306015ULL), UINT128(7258966386904442242ULL, 9036451851078271737ULL)},
//     {UINT128(2944506607763543303ULL, 17931860136402783984ULL), UINT128(12713825445334129788ULL, 8965715150203301741ULL), UINT128(4616365532043632928ULL, 9903947360375363881ULL), UINT128(1742220256128190082ULL, 16990745766680145097ULL), UINT128(2982956471696202352ULL, 7314716061732978286ULL), UINT128(4994602655911187373ULL, 11462958752220609541ULL), UINT128(14091557509287699902ULL, 2703592114389423683ULL), UINT128(15142154434585126153ULL, 1976066774697778237ULL), UINT128(8059710764380739076ULL, 13477513745589357156ULL), UINT128(4175777698439724595ULL, 7010464075490735258ULL), UINT128(11392909274603755801ULL, 4369547966950095062ULL), UINT128(3551243236294010821ULL, 12257840152266550034ULL), UINT128(1732946555642840329ULL, 14963345063076037644ULL), UINT128(6031862629015965663ULL, 771891216495984467ULL), UINT128(8443840983441100723ULL, 5239524175370117980ULL), UINT128(11944980128008973893ULL, 13470794821276171409ULL), UINT128(2642729473797562147ULL, 9946817225484376976ULL), UINT128(5102155788167780533ULL, 1867779596908554660ULL), UINT128(7388747831293255308ULL, 72218278911894728ULL), UINT128(10660017347289324442ULL, 9119631141775357588ULL), UINT128(8687473168003698916ULL, 14873834280937908355ULL), UINT128(7789245235632602840ULL, 11486104540641690964ULL), UINT128(9762070240410424496ULL, 13162756130524109217ULL), UINT128(10276049259224068516ULL, 11295885232880615728ULL), UINT128(9680954044651152481ULL, 16538454557724160982ULL)},
//     {UINT128(16990471171456519489ULL, 5628365462010493816ULL), UINT128(363651486045896965ULL, 4612057855179035013ULL), UINT128(5854708164919167204ULL, 7349570298716456357ULL), UINT128(12775048277511669531ULL, 5964432178155573120ULL), UINT128(3452242849270201043ULL, 10504621296780711200ULL), UINT128(56404473092493675ULL, 1723787334189489442ULL), UINT128(17536830473517522868ULL, 14697226705972549032ULL), UINT128(17105041422859695017ULL, 4609976411956699596ULL), UINT128(6617847938779528891ULL, 7497484614020213442ULL), UINT128(18443822972034614506ULL, 1007147733560765259ULL), UINT128(7736290335964753541ULL, 17311458528243287359ULL), UINT128(18266990238644617707ULL, 9536553771262595878ULL), UINT128(8205311123472786863ULL, 14219150876532409772ULL), UINT128(8560848045687112020ULL, 15111903683558143267ULL), UINT128(10681613158505319663ULL, 12952828330568256984ULL), UINT128(4897545808776209819ULL, 14850842625170639357ULL), UINT128(11211807681176949356ULL, 12461227379591454382ULL), UINT128(2990135542069244593ULL, 4682726586182886314ULL), UINT128(16872350852876971627ULL, 9739705020843384524ULL), UINT128(10729277725597588736ULL, 17442014543197543407ULL), UINT128(3281254259938057844ULL, 2169315633851905481ULL), UINT128(3192690396654436007ULL, 16415340874222909072ULL), UINT128(5775224991424894808ULL, 5884193111266793240ULL), UINT128(11895402785926878370ULL, 6308407939541610735ULL), UINT128(12911716826132515007ULL, 1567443591308202161ULL)},
//     {UINT128(4503483112682281731ULL, 17440346334246989491ULL), UINT128(16099987145904032363ULL, 14005119091644007893ULL), UINT128(6823741926835534565ULL, 16890985093688647694ULL), UINT128(3328600857453059143ULL, 17813236142531084219ULL), UINT128(2172816235672004541ULL, 8194719004162487122ULL), UINT128(15989169951077025146ULL, 10998653891990356681ULL), UINT128(2323303799822132324ULL, 5306740201644029308ULL), UINT128(6993864782032032845ULL, 11281030901473492784ULL), UINT128(13484324823643673312ULL, 3700791504705549229ULL), UINT128(7059938657736754235ULL, 14697212386384500679ULL), UINT128(3553481862455864898ULL, 6160944270859646903ULL), UINT128(6661094021063518203ULL, 5020009621503443629ULL), UINT128(3034123080167262035ULL, 17017907996537385270ULL), UINT128(18115204426267910464ULL, 1952111145328167565ULL), UINT128(2750396400825569938ULL, 2207325447887177153ULL), UINT128(13157043558772184064ULL, 7730875074565959441ULL), UINT128(7275383914450265285ULL, 16557201664583746815ULL), UINT128(10269734441447051692ULL, 10704675672324630250ULL), UINT128(12173560913344525673ULL, 14511642353320008925ULL), UINT128(2969543040806591028ULL, 16848543654513206978ULL), UINT128(11107515276915435324ULL, 4965292337848972618ULL), UINT128(11573386966675026912ULL, 4810572030441081219ULL), UINT128(8137641987281074510ULL, 14735997858011405115ULL), UINT128(5196750862630406027ULL, 9095723411436957778ULL), UINT128(803220617546047545ULL, 13553616297897430211ULL)},
//     {UINT128(3297034043278591588ULL, 8118542725501177227ULL), UINT128(8032561854409742349ULL, 2629752493665164554ULL), UINT128(14694093046672160829ULL, 8405318367441754292ULL), UINT128(9294693764776180080ULL, 10093301459969525436ULL), UINT128(9491938663112612168ULL, 11451487590333902832ULL), UINT128(7886192658063579987ULL, 8667188657554605114ULL), UINT128(18332100247151561556ULL, 17356309125185099828ULL), UINT128(2484969023278815960ULL, 12574036582508980802ULL), UINT128(5534472898170946684ULL, 11661174077163555328ULL), UINT128(11539082789171122486ULL, 7219096541336287652ULL), UINT128(12082323937184322144ULL, 11329219525861964312ULL), UINT128(556095610707405313ULL, 10614304292498858100ULL), UINT128(6493305072281843827ULL, 15293302170169929137ULL), UINT128(11958629760439883533ULL, 17691857377185323783ULL), UINT128(8403933754030667128ULL, 14504063871116522324ULL), UINT128(7879992289019938942ULL, 3716123971778324364ULL), UINT128(16231984608440033202ULL, 7125259028684661868ULL), UINT128(13490743115931184870ULL, 13502598755005102770ULL), UINT128(9300667661855667043ULL, 4022324185056109593ULL), UINT128(6840191725652544889ULL, 4118952125308336190ULL), UINT128(9779390840887924230ULL, 10652002770810544731ULL), UINT128(16599052193772265146ULL, 17223876769029730732ULL), UINT128(9767763165741647148ULL, 5740764827161805833ULL), UINT128(16536664333191975866ULL, 6311014577105986210ULL), UINT128(79421771652239086ULL, 6445664654160038011ULL)},
//     {UINT128(17906600100920952907ULL, 1696702158036383206ULL), UINT128(9097354538565255230ULL, 11257720490232638721ULL), UINT128(6845074864869734910ULL, 14863354650401757041ULL), UINT128(4264880636296826989ULL, 3012449169008094810ULL), UINT128(9492453997004047378ULL, 15564186969901516151ULL), UINT128(3783081589547560855ULL, 4412941021501932730ULL), UINT128(16374990460602430421ULL, 16003780765803343726ULL), UINT128(5779129833561564052ULL, 15424291874924839750ULL), UINT128(1204481080649999750ULL, 17739158017377128113ULL), UINT128(3900448616459014211ULL, 17973307099549099259ULL), UINT128(5571752286001313975ULL, 584642996609856800ULL), UINT128(7931627874594956809ULL, 9733163853829718833ULL), UINT128(12211241791986073436ULL, 5834510393874729820ULL), UINT128(3147922280748083519ULL, 17644202775123459686ULL), UINT128(16631852024963559521ULL, 2922431286034582067ULL), UINT128(14857081033695950408ULL, 8580457738106110709ULL), UINT128(6035740027471571453ULL, 7877370180151903808ULL), UINT128(10684283492854025776ULL, 15111581164596924479ULL), UINT128(13422956202979756577ULL, 11398862124590804593ULL), UINT128(9106162590582558824ULL, 2283154322307332344ULL), UINT128(11373175581093703498ULL, 977093936814760482ULL), UINT128(1609884136945291685ULL, 12568817897738185944ULL), UINT128(10641842875430115969ULL, 17320812210236094863ULL), UINT128(17492035678723885800ULL, 1243791834583853250ULL), UINT128(1018049409929183422ULL, 12004702730031156054ULL)},
//     {UINT128(13621160857800113927ULL, 17721411929140799088ULL), UINT128(195973734997869667ULL, 16022875907601438841ULL), UINT128(5087833880490287686ULL, 11720795229154712224ULL), UINT128(17526941859557199550ULL, 8999557934951375870ULL), UINT128(5938295485853664100ULL, 8923862537340921686ULL), UINT128(12655265883450907401ULL, 5555046601534899554ULL), UINT128(15766067301058693951ULL, 16961405260585422447ULL), UINT128(17115425721514075790ULL, 10266487654018719136ULL), UINT128(6969986483382063115ULL, 1182556844107041415ULL), UINT128(16843226032353691882ULL, 14687184656518502692ULL), UINT128(4273343246618372057ULL, 17616754038684144471ULL), UINT128(5675185733192417654ULL, 10922149291964392055ULL), UINT128(7021415831608451572ULL, 10034466978294824111ULL), UINT128(4553557440881127328ULL, 9686755736568951973ULL), UINT128(2513587731017102788ULL, 15235817736835184447ULL), UINT128(6307776766145679780ULL, 5377662972084577648ULL), UINT128(11982315151291367333ULL, 17031416962153772280ULL), UINT128(7184253244088690064ULL, 3648946628687402126ULL), UINT128(5919226746200821055ULL, 18195166310460279268ULL), UINT128(7799687608057401231ULL, 7177267547156926248ULL), UINT128(8982834928737795099ULL, 17234671348693906835ULL), UINT128(8546729897228840190ULL, 17668663682829029585ULL), UINT128(15278230290518519668ULL, 8003360191554296669ULL), UINT128(10524233768786539603ULL, 16360506997361265251ULL), UINT128(14700369557840508554ULL, 9339768956469115265ULL)},
//     {UINT128(14730934658419703433ULL, 11196748959976794551ULL), UINT128(8877941919806554809ULL, 988008284332548924ULL), UINT128(15040877649923759664ULL, 15464666722459585081ULL), UINT128(14713965842558099199ULL, 17837493944427475027ULL), UINT128(10291055317222324882ULL, 3529488088159729638ULL), UINT128(17334377365388475457ULL, 4215380886305619012ULL), UINT128(17148571958833633919ULL, 6145393677406687977ULL), UINT128(15811292644143325070ULL, 12489567810906135806ULL), UINT128(8939534736424067502ULL, 8510060388039597086ULL), UINT128(6026483479143217351ULL, 5102638719895662566ULL), UINT128(6940234984529767770ULL, 14333985850450410498ULL), UINT128(6467422054080500551ULL, 2983632514257750884ULL), UINT128(8840000277830708982ULL, 2175813522678069768ULL), UINT128(4292302085637321267ULL, 18176448287043232527ULL), UINT128(10698686279447508477ULL, 16733245294171198792ULL), UINT128(4873350868608581078ULL, 14381976274356205764ULL), UINT128(12194785184974896534ULL, 6823370275790663769ULL), UINT128(10762355975940314225ULL, 13136358775303289843ULL), UINT128(8753949718101195035ULL, 11236959591511980918ULL), UINT128(7633931121317849006ULL, 11247955699933111686ULL), UINT128(10684090810307235502ULL, 3803718197250203926ULL), UINT128(15183143464562300447ULL, 12210997929165604385ULL), UINT128(8486965043110653466ULL, 10373343045062075370ULL), UINT128(16553252376064689344ULL, 1072331951408202852ULL), UINT128(11252407941020339916ULL, 4359337434540924586ULL)},
//     {UINT128(1827479429594962343ULL, 8707554268680212376ULL), UINT128(11731382104168280276ULL, 15480069734044620408ULL), UINT128(5114056441191457400ULL, 11222617921715687272ULL), UINT128(7574851068794621054ULL, 10055540579295186083ULL), UINT128(16832522178247739505ULL, 16438151509325860910ULL), UINT128(12031568742176668968ULL, 388168453214079737ULL), UINT128(13376814459111167495ULL, 2737105890626629792ULL), UINT128(5666980697243822790ULL, 1695011218686494555ULL), UINT128(12714860612609914706ULL, 3636866279592794143ULL), UINT128(7960345742285168892ULL, 4589563965076997262ULL), UINT128(14011147076667953797ULL, 12699602845968426827ULL), UINT128(8114077333086473890ULL, 5365062577327467559ULL), UINT128(7644272578021081187ULL, 417364961761106115ULL), UINT128(5301571030495055934ULL, 10177904325818183647ULL), UINT128(3822524227704124309ULL, 2631643967634088960ULL), UINT128(13507096276508960632ULL, 8912856112655254754ULL), UINT128(10319366238741038338ULL, 17617724083507453722ULL), UINT128(10741153480536028283ULL, 8055001269525456455ULL), UINT128(13952180610846350671ULL, 372199128990115073ULL), UINT128(8642633810008999290ULL, 10571604239150408097ULL), UINT128(7659733159053448049ULL, 16124100676642183591ULL), UINT128(17499410814980450660ULL, 12800406879094780923ULL), UINT128(10229240684557221152ULL, 11067817983867839092ULL), UINT128(14388921905362393749ULL, 6374518056561694498ULL), UINT128(10498629036311646030ULL, 2868516815507340681ULL)},
//     {UINT128(17396061356597461327ULL, 3661304287894344073ULL), UINT128(2157479581379360706ULL, 2964429873997552845ULL), UINT128(6537573419625436066ULL, 5846621520790178159ULL), UINT128(16415911896056318860ULL, 15562069518014971803ULL), UINT128(7363518214551560680ULL, 6819737876243505845ULL), UINT128(695355510427057623ULL, 8233499156835938828ULL), UINT128(7970534935915557267ULL, 9306392511994419423ULL), UINT128(11974277327068721133ULL, 9950195621826024579ULL), UINT128(17597052086179375985ULL, 205427416734991704ULL), UINT128(13322959768238634042ULL, 5621137227441516111ULL), UINT128(16702290331262006731ULL, 14018210243993021163ULL), UINT128(2078678950530705897ULL, 7715161545317547635ULL), UINT128(15064418241677348838ULL, 6569305926353661436ULL), UINT128(1040329313929484267ULL, 23052565181311149ULL), UINT128(3936722852794617887ULL, 7132395294159041885ULL), UINT128(3932224325456971280ULL, 13857902073478665796ULL), UINT128(8854628408407799776ULL, 9922992214893918196ULL), UINT128(9384035787036274215ULL, 897427079418034470ULL), UINT128(3052019359749308592ULL, 1498168202329104695ULL), UINT128(4597032401938108309ULL, 4083209445547746280ULL), UINT128(13658329125297853528ULL, 16072854639120525947ULL), UINT128(15021240615246256319ULL, 6224558340371235945ULL), UINT128(3249029594869983928ULL, 4934056985448454730ULL), UINT128(15632987706839527221ULL, 44313963877018103ULL), UINT128(10779812691781178345ULL, 1312248877787571989ULL)},
//     {UINT128(11757949057358720375ULL, 13728411539020162312ULL), UINT128(14458192145444511498ULL, 1492611583268731797ULL), UINT128(7510081255081831661ULL, 6724942000368971681ULL), UINT128(7302017187585711121ULL, 1087515111743675020ULL), UINT128(7031995978617174842ULL, 14092310361577302850ULL), UINT128(11608877918319000581ULL, 7064506195439588468ULL), UINT128(16564582501842002561ULL, 16216022598163343139ULL), UINT128(312728523455662856ULL, 12261846317063337038ULL), UINT128(106828003810215965ULL, 8685965625949269099ULL), UINT128(2124413591913142981ULL, 11780753130074411157ULL), UINT128(18315355853608097105ULL, 17425152415764704082ULL), UINT128(15889972037832956478ULL, 6362495723982864933ULL), UINT128(6326394029689545518ULL, 3793130991675087962ULL), UINT128(7220513307983432157ULL, 13116854574210097098ULL), UINT128(1992229418553602812ULL, 2977111041307276303ULL), UINT128(6806411781253690035ULL, 3213613201974644795ULL), UINT128(12222676252211750180ULL, 6158303037007999144ULL), UINT128(18131829029827715413ULL, 18097943068155809186ULL), UINT128(14038641570810519459ULL, 2708408313265576994ULL), UINT128(17078092375003363132ULL, 10669842456614527817ULL), UINT128(11578824537142057661ULL, 3919235633219041889ULL), UINT128(14701341572152225922ULL, 658654837695871204ULL), UINT128(2936288253089703377ULL, 14950366102210864585ULL), UINT128(4886524843396071135ULL, 16882972221608864899ULL), UINT128(17879892427803806945ULL, 14510370429694874078ULL)}
// };

// const __uint128_t z_color[2] = {
//     UINT128(10328872305587225489ULL, 9724626366735152691ULL),
//     UINT128(8532565151760883153ULL, 14982123156822918432ULL)
// };


#endif