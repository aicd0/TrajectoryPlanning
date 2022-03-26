#include <ros/ros.h>
#include <sdf/sdf.hh>
#include <boost/bind/bind.hpp>
#include "gazebo/gazebo.hh"
#include "gazebo/common/Plugin.hh"
#include "gazebo/msgs/msgs.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/transport/transport.hh"
#include "robot_sim/StepWorldSrvInfo.h"

namespace gazebo
{
  class StepWorldPlugin : public WorldPlugin
  {
  private:
    ros::NodeHandle m_node_handle;
    ros::ServiceServer m_service;
    physics::WorldPtr m_world_ptr;

  public:
    StepWorldPlugin() :
      m_node_handle("step_control_plugin") {}

    void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
    {
      if (!ros::isInitialized())
      {
        ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin.");
        return;
      }

      m_world_ptr = _world;

      // Create ROS service.
      m_service = m_node_handle.advertiseService
      <robot_sim::StepWorldSrvInfo::Request, robot_sim::StepWorldSrvInfo::Response>(
        "/user/step_world",
        boost::bind(&StepWorldPlugin::step_srv_callback, this, boost::placeholders::_1, boost::placeholders::_2)
      );

      // Pause the world.
      m_world_ptr->SetPaused(true);
    }

  private:
    bool step_srv_callback(
      robot_sim::StepWorldSrvInfo::Request  &req,
      robot_sim::StepWorldSrvInfo::Response &res)
    {
      res.old_iterations = m_world_ptr->Iterations();
      m_world_ptr->Step(req.steps);
      res.new_iterations = m_world_ptr->Iterations();
      return true;
    }
  };

  GZ_REGISTER_WORLD_PLUGIN(StepWorldPlugin)
}
