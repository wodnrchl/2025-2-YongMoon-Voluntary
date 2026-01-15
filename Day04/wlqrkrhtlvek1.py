import numpy as np

class PongEnv:
    def __init__(self, render_mode=None, target_fps = 60):
        """
        환경 초기화
        
        Args:
            render_mode: 'human'이면 화면 렌더링, None이면 헤드리스 모드
        """
        self.width = 800
        self.height = 600

        self.ball_x = None
        self.ball_y = None
        self.ball_dx = None
        self.ball_dy = None

        self.paddle_x = None  # 패들 가로 위치 (0~1)
        self.paddle_width = 0.7  # 패들 너비 (화면 대비 비율)
        
        # 점수 추가
        self.score = 0
        # 놓친 횟수 추적
        self.misses = 0

        # 액션 스페이스와 관측 스페이스 정의
        self.action_space_n = 3  # 0: 왼쪽, 1: 정지, 2: 오른쪽
        self.observation_space_shape = (5,)
        
        # 렌더링 설정
        self.render_mode = render_mode
        self.target_fps = 120  # FPS 증가 (기존 60)

        # 공과 패들 속도 설정
        self.ball_speed_min = 0.025 # 시작 속도
        self.ball_speed_max = 0.3  # 최대 속도
        self.paddle_speed_min = 0.0125
        self.paddle_speed_max = 0.05
        self.target_fps = 120

        # 두 번째 공 상태 변수
        self.ball2_x = None
        self.ball2_y = None
        self.ball2_dx = None
        self.ball2_dy = None
        self.ball2_active = False  # 두 번째 공 활성화 여부
        self.ball2_delay = 2  # 두 번째 공 등장 딜레이(초)
        self.ball2_timer = 0.0

        # 빨간 공 상태 변수 (여러 개)
        #self.red_balls = []  # 각 빨간 공: {'x':..., 'y':..., 'active':...}
        #self.red_ball_speed = 0.01
        #self.red_ball_interval = 5  # 5점 간격 등장
        #self.red_ball_start_score = 15

        # 화면 렌더링 변수
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        if render_mode == 'human':
            try:
                import pygame
                pygame.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Pong 게임")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 74)
                self.small_font = pygame.font.Font(None, 48)
            except ImportError:
                print("pygame이 설치되지 않았습니다. headless mode로 실행하거나 pygame을 설치하세요.")
                self.render_mode = None

    def reset(self):
        """환경을 초기 상태로 리셋"""
        self.ball_x = 0.5  # 화면 중앙
        self.ball_y = 0.1  # 화면 상단
        self.paddle_x = 0.5  # 패들 중앙
        self.ball_dx = np.random.choice([-self.ball_speed_min, self.ball_speed_min])  # 좌우 랜덤
        self.ball_dy = self.ball_speed_min  # 아래로 이동
        self.current_ball_speed = self.ball_speed_min
        self.current_paddle_speed = self.paddle_speed_min
        
        # 두 번째 공 초기화 (딜레이 적용)
        self.ball2_x = 0.5
        self.ball2_y = 0.2
        self.ball2_dx = np.random.choice([-self.ball_speed_min, self.ball_speed_min])
        self.ball2_dy = self.ball_speed_min
        self.ball2_active = False
        self.ball2_timer = 0.0
        self.ball1_missed = False
        self.ball2_missed = False

        self.score = 0
        self.misses = 0

        # 빨간 공 초기화
        self.red_balls = []

        return self._get_state()

    def step(self, action):
        """
        action:
        0 = 왼쪽으로 이동
        1 = 정지
        2 = 오른쪽으로 이동

        반환값:
        state : 현재 상태
        reward : 보상
        done : 에피소드 종료 여부
        info : 추가 정보 (딕셔너리)
        """
        # 점수에 따라 속도 증가 (최대 1000000점까지 선형 증가, 증가폭 완만하게)
        speed_ratio = min(self.score / 1000000, 1.0)  # 1000000점까지 선형 증가
        self.current_ball_speed = self.ball_speed_min + (self.ball_speed_max - self.ball_speed_min) * speed_ratio
        self.current_paddle_speed = self.paddle_speed_min + (self.paddle_speed_max - self.paddle_speed_min) * speed_ratio
        # 방향 유지
        self.ball_dx = np.sign(self.ball_dx) * self.current_ball_speed
        self.ball_dy = np.sign(self.ball_dy) * self.current_ball_speed
        self.ball2_dx = np.sign(self.ball2_dx) * self.current_ball_speed
        self.ball2_dy = np.sign(self.ball2_dy) * self.current_ball_speed

        # 두 번째 공 딜레이 처리
        import time
        if not self.ball2_active:
            self.ball2_timer += 1.0 / self.target_fps
            if self.ball2_timer >= self.ball2_delay:
                self.ball2_active = True

        if action == 0:
            self.paddle_x -= self.current_paddle_speed
        elif action == 2:
            self.paddle_x += self.current_paddle_speed
        
        # 패들이 화면 밖으로 나가지 않도록 제한
        self.paddle_x = np.clip(self.paddle_x, 0.0, 1.0)

        # 공 이동 로직 구현
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        if self.ball2_active:
            self.ball2_x += self.ball2_dx
            self.ball2_y += self.ball2_dy
        
        reward = 0.0
        done = False

        # 좌우 벽 충돌 구현
        if self.ball_x <= 0.0 or self.ball_x >= 1.0:
            self.ball_dx *= -1
        if self.ball2_active and (self.ball2_x <= 0.0 or self.ball2_x >= 1.0):
            self.ball2_dx *= -1
        
        # 위쪽 벽 충돌 구현
        if self.ball_y <= 0.0:
            self.ball_dy *= -1
        if self.ball2_active and self.ball2_y <= 0.0:
            self.ball2_dy *= -1

        # 패들 충돌 및 점수 시스템 구현
        if self.ball_y >= 0.95 and not self.ball1_missed:
            if abs(self.ball_x - self.paddle_x) <= self.paddle_width / 2:
                self.ball_dy *= -1
                # x방향 랜덤화
                self.ball_dx = np.random.choice([-1, 1]) * abs(self.ball_dx)
                self.score += 1
                reward += 1.0
                # 10000점 초과 시 0.001% 확률로 즉시 게임오버
                if self.score > 10000:
                    if np.random.rand() < 0.000001:
                        done = True
            else:
                self.misses += 1
                reward -= 1.0
                self.ball1_missed = True

        if self.ball2_active and self.ball2_y >= 0.95 and not self.ball2_missed:
            if abs(self.ball2_x - self.paddle_x) <= self.paddle_width / 2:
                self.ball2_dy *= -1
                # x방향 랜덤화
                self.ball2_dx = np.random.choice([-1, 1]) * abs(self.ball2_dx)
                self.score += 1
                reward += 1.0
                # 10000점 초과 시 0.001% 확률로 즉시 게임오버
                if self.score > 10000:
                    if np.random.rand() < 0.000001:
                        done = True
            else:
                self.misses += 1
                reward -= 1.0
                self.ball2_missed = True

        # 두 공 모두 놓쳐야 게임오버
        if self.ball1_missed and self.ball2_missed:
            done = True

        # info 딕셔너리
        info = {
            'score': self.score,
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        """
        AI에게 줄 입력값(관측값) 추출
        현재 상태를 numpy 배열로 반환
        
        반환할 값들:
        - self.ball_x: 공의 x 위치
        - self.ball_y: 공의 y 위치
        - self.paddle_x: 패들의 x 위치
        - self.ball_dx: 공의 x 방향 속도
        - self.ball_dy: 공의 y 방향 속도
        
        hint: np.array([값1, 값2, ...], dtype=np.float32)
        """
        return np.array([
             self.ball_x,
             self.ball_y,
             self.ball2_x,
             self.ball2_y,
             self.paddle_x,
             self.ball_dx,
             self.ball_dy,
             self.ball2_dx,
             self.ball2_dy,
             float(self.ball2_active),
             ], dtype= np.float32)


    # 화면 렌더링 관련 함수
    def render(self):
        """
        게임 화면 렌더링 (render_mode='human'일 때만 작동)
        """
        if self.render_mode != 'human' or self.screen is None:
            return
        
        import pygame
        
        # 배경
        self.screen.fill((0, 0, 0))
        
        # 패들 (아래쪽, 가로로)
        paddle_pixel_x = int(self.paddle_x * self.width)
        paddle_pixel_width = int(self.paddle_width * self.width)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (paddle_pixel_x - paddle_pixel_width // 2, self.height - 30,
                         paddle_pixel_width, 18))
        
        # 공
        ball_pixel_x = int(self.ball_x * self.width)
        ball_pixel_y = int(self.ball_y * self.height)
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (ball_pixel_x - 7, ball_pixel_y - 7, 15, 15))
        
        # 두 번째 공 (다른 색, 활성화 시만)
        if self.ball2_active:
            ball2_pixel_x = int(self.ball2_x * self.width)
            ball2_pixel_y = int(self.ball2_y * self.height)
            pygame.draw.rect(self.screen, (0, 255, 255),
                            (ball2_pixel_x - 7, ball2_pixel_y - 7, 15, 15))
        
        # 빨간 공 렌더링
        for ball in self.red_balls:
            if ball['active']:
                ball_pixel_x = int(ball['x'] * self.width)
                ball_pixel_y = int(ball['y'] * self.height)
                pygame.draw.circle(self.screen, (255, 0, 0), (ball_pixel_x, ball_pixel_y), 10)
        
        # 점수 표시
        score_text = self.font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (self.width // 2 - 30, 50))
        
        # Miss 횟수 표시
        #miss_text = self.small_font.render(f"Miss: {self.misses}", True, (255, 255, 255))
        #self.screen.blit(miss_text, (self.width - 150, 20))

        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(self.target_fps)
    
    def render_game_over(self):
        """게임 오버 화면 렌더링"""
        if self.render_mode != 'human' or self.screen is None:
            return None
        
        import pygame
        
        # 반투명 오버레이
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # GAME OVER 텍스트
        game_over_font = pygame.font.Font(None, 120)
        game_over_text = game_over_font.render("GAME OVER", True, (255, 50, 50))
        text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2 - 100))
        self.screen.blit(game_over_text, text_rect)
        
        # 최종 점수 표시
        score_font = pygame.font.Font(None, 60)
        score_text = score_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 - 20))
        self.screen.blit(score_text, score_rect)
        
        # 다시 시작 버튼
        button_width = 250
        button_height = 60
        button_x = self.width // 2 - button_width // 2
        button_y = self.height // 2 + 50
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        
        # 마우스 호버 체크
        mouse_pos = pygame.mouse.get_pos()
        is_hover = button_rect.collidepoint(mouse_pos)
        
        # 버튼 모양
        button_color = (100, 255, 100) if is_hover else (50, 200, 50)
        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=15)
        pygame.draw.rect(self.screen, (255, 255, 255), button_rect, 4, border_radius=15)
        
        # 버튼 텍스트
        button_font = pygame.font.Font(None, 48)
        button_text = button_font.render("RESTART", True, (0, 0, 0))
        button_text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, button_text_rect)
        
        pygame.display.flip()
        
        return button_rect

    def close(self):
        """환경 종료 및 리소스 정리"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None


# ====================================================================
# 테스트 코드
# ====================================================================
if __name__ == "__main__":
    import pygame
    
    # 렌더링 모드로 환경 생성
    env = PongEnv(render_mode='human')
    
    # 게임 초기화
    state = env.reset()
    print(f"초기 상태: {state}")
    print(f"상태 형태: {state.shape}")
    print(f"액션 스페이스: {env.action_space_n}")
    print("\n조작법: 왼쪽 화살표=왼쪽, 오른쪽 화살표=오른쪽")
    
    # 게임 플레이 (키보드 조작)
    running = True
    total_reward = 0
    steps = 0
    action = 1  # 초기 액션: 정지
    
    while running:
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 키보드 입력 받기
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0  # 왼쪽
        elif keys[pygame.K_RIGHT]:
            action = 2  # 오른쪽
        else:
            action = 1  # 정지
        
        # 스텝 실행
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # 렌더링
        env.render()
        
        # 상태 정보 출력
        if steps % 100 == 0:
            print(f"Steps: {steps}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"Score: {info['score']}")
        
        # 게임 종료 시 리셋
        if done:
            # GAME OVER 화면 표시 및 버튼 대기
            waiting = True
            while waiting:
                button_rect = env.render_game_over()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    # 마우스 클릭 체크
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if button_rect and button_rect.collidepoint(event.pos):
                            waiting = False
            
            print(f"\n게임 종료! 총 스텝: {steps}, 총 보상: {total_reward:.2f}")
            print(f"최종 점수 - Score: {info['score']}")
            state = env.reset()
            total_reward = 0
            steps = 0
    
    env.close()
