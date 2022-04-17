#include <ros/ros.h>
#include <sdf/sdf.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include "robot_sim/StepWorld.h"
#include "time_estimator.h"

namespace gazebo
{
  class StepWorldPlugin : public WorldPlugin
  {
  private:
    // ros
    ros::NodeHandle m_ros_nh;
    ros::ServiceServer m_service;

    // gazebo
    gazebo::transport::NodePtr m_gazebo_node;
    gazebo::transport::PublisherPtr m_wc_pub;
    physics::WorldPtr m_world_ptr;

    // misc
    TimeEstimator m_call_time_estimator = 50U;

  public:
    StepWorldPlugin() :
      m_ros_nh("step_world_plugin"),
      m_gazebo_node(new gazebo::transport::Node())
    {
      m_gazebo_node->Init();
    }

    void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      if (!ros::isInitialized())
      {
        ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin.");
        return;
      }

      m_world_ptr = _world;

      // Pause the world.
      m_world_ptr->SetPaused(true);
  
      // Advertise gazebo publisher.
      m_wc_pub = m_gazebo_node->Advertise<gazebo::msgs::WorldControl>("~/world_control");
      m_wc_pub->WaitForConnection();

      // Create ROS service.
      m_service = m_ros_nh.advertiseService
      <robot_sim::StepWorld::Request, robot_sim::StepWorld::Response>(
        "/user/step_world",
        std::bind(&StepWorldPlugin::step_srv_callback, this, std::placeholders::_1, std::placeholders::_2)
      );
    }

  private:
    bool step_srv_callback(
      robot_sim::StepWorld::Request  &req,
      robot_sim::StepWorld::Response &res)
    {
      res.old_iterations = m_world_ptr->Iterations();
      
      // Publish stepper.
      gazebo::msgs::WorldControl stepper;
      stepper.set_multi_step(req.steps);
      m_wc_pub->Publish(stepper);

      for (;;) {
        unsigned int commit = m_call_time_estimator.remain();
        std::this_thread::sleep_for(std::chrono::milliseconds(commit));
        bool completed = m_world_ptr->Iterations() >= res.old_iterations + req.steps;
        m_call_time_estimator.commit(commit, completed);
        if (completed) {
          break;
        }
      }
      
      res.new_iterations = m_world_ptr->Iterations();
      return true;
    }
  };

  GZ_REGISTER_WORLD_PLUGIN(StepWorldPlugin)
}
