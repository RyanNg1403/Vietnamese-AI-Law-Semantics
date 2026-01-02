% =======================================================================
% FILE: rules.pl
% MÔ TẢ: Hệ luật logic mô tả các định nghĩa trong Điều 3 Luật AI
% =======================================================================

% --- PHẦN 1: NẠP KIẾN THỨC NỀN TỪ PYTHON ---
% Nạp file facts mà bạn đã tạo ở bước trước
:- consult('knowledge_base.pl').

% Khai báo các vị từ động để tránh lỗi nếu dữ liệu chưa có
:- dynamic organization/1, individual/1, system/1, ai_system/1.
:- dynamic designs/2, builds/2, trains/2, brings_to_market/2, uses/2.
:- dynamic event/1, causes_damage/2.
:- dynamic machine_based/1, has_autonomy/2, performs_ai_capabilities/1.
:- dynamic artificial_intelligence_related/1.

% =======================================================================
% --- PHẦN 2: DỮ LIỆU GIẢ LẬP (MOCK DATA) ĐỂ KIỂM THỬ ---
% Phần này tạo ra các đối tượng cụ thể để test xem luật chạy đúng không.
% Bạn có thể thêm bớt tùy ý.
% =======================================================================

% 1. Định nghĩa các thực thể (Entities)
organization(bkav_corp).        % BKAV là tổ chức
organization(fpt_software).     % FPT là tổ chức
individual(nguyen_van_an).      % Nguyễn Văn An là cá nhân
individual(le_thi_b).           % Lê Thị B là cá nhân
state_agency(bo_cong_an).       % Bộ Công An là cơ quan nhà nước

% 2. Định nghĩa các hệ thống (Systems)
system(chat_gpt_vn).            % ChatGPT VN là hệ thống
system(camera_traffic).         % Camera giao thông là hệ thống
system(excel_macro).            % Excel Macro là hệ thống thường, không phải AI
% Note: ai_system/1 được định nghĩa bằng rule (xem PHẦN 3)

% 2.4. Thuộc tính của hệ thống AI (để hỗ trợ định nghĩa)
machine_based(chat_gpt_vn).     % ChatGPT VN là hệ thống dựa trên máy
machine_based(camera_traffic).  % Camera giao thông là hệ thống dựa trên máy
machine_based(excel_macro).     % Excel Macro cũng là hệ thống dựa trên máy

has_autonomy(chat_gpt_vn, high).        % ChatGPT VN có mức độ tự chủ cao
has_autonomy(camera_traffic, medium).   % Camera có mức độ tự chủ trung bình
has_autonomy(excel_macro, low).         % Excel Macro có mức độ tự chủ thấp

performs_ai_capabilities(chat_gpt_vn).      % ChatGPT VN thực hiện khả năng AI
performs_ai_capabilities(camera_traffic).   % Camera thực hiện khả năng AI
% excel_macro KHÔNG có performs_ai_capabilities

% 3. Định nghĩa các hành động/quan hệ (Actions/Relations)
% BKAV thiết kế và huấn luyện ChatGPT VN
designs(bkav_corp, chat_gpt_vn).
trains(bkav_corp, chat_gpt_vn).

% FPT đưa Camera giao thông ra thị trường
brings_to_market(fpt_software, camera_traffic).

% Ông An sử dụng ChatGPT VN cho mục đích cá nhân
uses(nguyen_van_an, chat_gpt_vn).
purpose(nguyen_van_an, chat_gpt_vn, personal).

% Bộ Công An sử dụng Camera giao thông cho mục đích công vụ
uses(bo_cong_an, camera_traffic).
purpose(bo_cong_an, camera_traffic, professional).

% Một sự cố xảy ra: Camera nhận diện sai gây thiệt hại danh dự
event(incident_01).
occurs_in(incident_01, camera_traffic).
causes_damage(incident_01, reputation). 

% =======================================================================
% --- PHẦN 3: CÁC LUẬT SUY DIỄN (LOGICAL RULES) ---
% Dựa trên Điều 3 của văn bản luật 
% =======================================================================

% -----------------------------------------------------------------------
% ĐỊNH NGHĨA 0: ARTIFICIAL INTELLIGENCE (Trí tuệ nhân tạo) - Câu 1
% Luật: Là sự triển khai điện tử của khả năng trí tuệ con người
% Note: Đây là định nghĩa khái niệm, không phải role để truy vấn trực tiếp
% -----------------------------------------------------------------------
% artificial_intelligence được hiểu là khái niệm tổng hợp
% Có thể kiểm tra qua các concepts liên quan: intelligence, capability, implementation
artificial_intelligence_related(Concept) :-
    is_concept(Concept),
    (   Concept = intelligence
    ;   Concept = capability
    ;   Concept = implementation
    ;   has_synset(Concept, _)
    ).

% -----------------------------------------------------------------------
% ĐỊNH NGHĨA 0.5: AI SYSTEM (Hệ thống AI) - Câu 2
% Luật: Là hệ thống dựa trên máy được thiết kế để thực hiện khả năng AI 
%       với các mức độ tự chủ khác nhau
% -----------------------------------------------------------------------
ai_system(System) :-
    system(System),
    machine_based(System),
    performs_ai_capabilities(System),
    has_autonomy(System, _Level).  % Có mức độ tự chủ (bất kỳ level nào)

% -----------------------------------------------------------------------
% ĐỊNH NGHĨA 1: DEVELOPER (Nhà phát triển) - Khoản 3 Điều 3 [cite: 13]
% Luật: Là Tổ chức/Cá nhân + (Thiết kế HOẶC Xây dựng HOẶC Huấn luyện...) + Hệ thống AI.
% -----------------------------------------------------------------------
is_developer(Actor, System) :-
    (organization(Actor) ; individual(Actor)),    % Actor là tổ chức hoặc cá nhân
    ai_system(System),                            % System phải là AI
    (   designs(Actor, System)
    ;   builds(Actor, System)
    ;   trains(Actor, System)
    ;   tests(Actor, System)
    ;   fine_tunes(Actor, System)
    ).

% -----------------------------------------------------------------------
% ĐỊNH NGHĨA 2: PROVIDER (Nhà cung cấp) - Khoản 4 Điều 3 [cite: 14]
% Luật: Là Tổ chức/Cá nhân + Đưa ra thị trường HOẶC Đưa vào sử dụng + tên thương hiệu mình.
% -----------------------------------------------------------------------
is_provider(Actor, System) :-
    (organization(Actor) ; individual(Actor)),
    ai_system(System),
    (   brings_to_market(Actor, System)
    ;   puts_into_service(Actor, System)
    ).

% -----------------------------------------------------------------------
% ĐỊNH NGHĨA 3: DEPLOYER (Người triển khai) - Khoản 5 Điều 3 [cite: 15, 16]
% Luật: Là Tổ chức/Cá nhân/Cơ quan NN + Sử dụng AI + Mục đích chuyên môn/thương mại
% (LOẠI TRỪ mục đích cá nhân phi thương mại).
% -----------------------------------------------------------------------
is_deployer(Actor, System) :-
    (organization(Actor) ; individual(Actor) ; state_agency(Actor)),
    ai_system(System),
    uses(Actor, System),
    % Kiểm tra mục đích sử dụng (Logic phủ định)
    purpose(Actor, System, Purpose),
    Purpose \= personal. % Không phải là cá nhân (Non-commercial check)

% -----------------------------------------------------------------------
% ĐỊNH NGHĨA 4: USER (Người dùng) - Khoản 6 Điều 3 [cite: 17]
% Luật: Là Tổ chức/Cá nhân + Tương tác trực tiếp HOẶC Sử dụng đầu ra.
% -----------------------------------------------------------------------
is_user(Actor, System) :-
    (organization(Actor) ; individual(Actor)),
    ai_system(System),
    (   interacts_directly(Actor, System)
    ;   uses(Actor, System)
    ).

% -----------------------------------------------------------------------
% ĐỊNH NGHĨA 5: SERIOUS INCIDENT (Sự cố nghiêm trọng) - Khoản 8 Điều 3 [cite: 19]
% Luật: Sự kiện xảy ra trong hệ thống AI + Gây hại (Sức khỏe, tính mạng, tài sản...).
% -----------------------------------------------------------------------
is_serious_incident(Event) :-
    event(Event),
    occurs_in(Event, System),
    ai_system(System),
    causes_damage(Event, Type),
    % Danh sách các loại thiệt hại nghiêm trọng theo luật
    member(Type, [human_life, health, property, reputation, national_security, environment]).

% =======================================================================
% HỖ TRỢ SUY DIỄN TỪ WORDNET (Nếu file Python đã tạo facts sub_class)
% =======================================================================
% Luật bắc cầu: Nếu X là Developer, và Developer là Person (từ WordNet), thì X là Person.
is_a(Object, Class) :- 
    sub_class(Object, Class).
    
is_a(Object, GrandParent) :- 
    sub_class(Object, Parent), 
    is_a(Parent, GrandParent).